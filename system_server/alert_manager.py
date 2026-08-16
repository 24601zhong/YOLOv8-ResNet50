# -*- coding: utf-8 -*-
"""
异常预警触发模块 alert_manager.py
==================================
功能：
  1. 预警触发：前端弹窗 + 本地音频蜂鸣提示
  2. 截取当前监控帧保存至 output/alert_screenshots
  3. 将预警信息写入 alert_log 数据表
独立调试标准：手动输入低相似度特征可完整触发三项动作
"""

import os
import sys
import time
import json
import threading
import queue
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

import cv2


class AlertManager:
    """
    异常预警管理器
    - 预警触发（弹窗 + 蜂鸣）
    - 截图保存
    - 日志写入
    - 按人聚类（同一异常人员复用同一 person_key, 人脸优先 + ReID 兜底）
    """

    # 同人判定余弦阈值 (特征已 L2 归一化, 点积即余弦)
    FACE_CLUSTER_THRESHOLD = 0.5   # 人脸 512 维
    REID_CLUSTER_THRESHOLD = 0.7   # ReID 2048 维

    def __init__(self, output_dir=None, db_config=None):
        """
        :param output_dir: 截图保存目录
        :param db_config: MySQL 数据库配置
        """
        # 截图输出目录
        if output_dir is None:
            output_dir = Path(__file__).parent.parent / 'output' / 'alert_screenshots'
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 数据库配置
        self.db_config = db_config or {
            'host': 'localhost',
            'port': 3306,
            'user': 'root',
            'password': '123456',
            'database': 'hotel_security'
        }

        # 预警队列（用于异步处理）
        self.alert_queue = queue.Queue(maxsize=100)

        # 预警记录
        self.alert_history = []

        # 蜂鸣频率和时长
        self.beep_frequency = 800  # Hz
        self.beep_duration = 200   # ms

        # 统计
        self.stats = {
            'total_alerts': 0,
            'screenshots_saved': 0,
            'db_writes': 0,
            'beeps_triggered': 0,
            'last_alert_time': None
        }

        # 蜂鸣冷却（防止频繁触发）
        self.beep_cooldown = 3.0  # 秒
        self.last_beep_time = 0

        # 弹窗冷却
        self.popup_cooldown = 5.0  # 秒
        self.last_popup_time = 0

        # 预警冷却（防止同一摄像头每帧都触发完整预警, 拖垮检测循环）
        self.alert_cooldown = 5.0  # 秒
        self._last_alert_time = {}  # (camera_id, person_key) -> 上次触发时间戳

        # 按人聚类状态: person_key -> {'face_vec': 512维人脸向量|None, 'reid_vec': 2048维全身向量|None}
        self._clusters = {}
        self._cluster_seq = 0  # 自增序号, 保证新 key 不重复

        # 从已有预警记录重建聚类 (重启后同一人仍归入同一 person_key)
        self._rebuild_clusters()

        print(f"[INFO] 预警管理器初始化完成")
        print(f"[INFO] 截图保存目录: {self.output_dir}")

    def trigger_alert(self, camera_id, frame, similarity,
                      person_id=-1, person_info=None,
                      feature_vec=None, feature_type=None,
                      face_feature_vec=None, reid_feature_vec=None):
        """
        触发预警（完整流程）
        :param camera_id: 摄像头编号
        :param frame: 当前监控帧（BGR）
        :param similarity: 匹配相似度
        :param person_id: 匹配人员ID（-1 表示未匹配）
        :param person_info: 人员信息
        :param feature_vec: 首选特征向量 (人脸 512 / ReID 2048), 用于入库与后续登记
        :param feature_type: 'face' 或 'reid'
        :param face_feature_vec: 人脸特征 (512, 检出脸才有), 用于人脸簇
        :param reid_feature_vec: ReID 全身特征 (2048, 始终有), 用于 ReID 簇
        :return: 预警记录字典
        """
        # 按人聚类: 分配/复用 person_key (人脸优先 + ReID 兜底, 且 face↔reid 跨类型合并)
        face_vec = (face_feature_vec if face_feature_vec is not None
                    else (feature_vec if feature_type == 'face' else None))
        reid_vec = (reid_feature_vec if reid_feature_vec is not None
                    else (feature_vec if feature_type == 'reid' else None))
        person_key = self._assign_person_key(face_vec, reid_vec)

        # 预警冷却: 按 (摄像头, 人) 维度, 同人冷却期内不重复触发, 不同人各自立即触发
        now = time.time()
        cooldown_key = (camera_id, person_key)
        if now - self._last_alert_time.get(cooldown_key, 0) < self.alert_cooldown:
            return None
        self._last_alert_time[cooldown_key] = now

        alert_time = datetime.now()

        # Step 1: 保存截图
        screenshot_path = self._save_screenshot(frame, camera_id, alert_time)

        # Step 2: 触发蜂鸣
        self._play_beep()

        # Step 3: 触发弹窗
        alert_popup_info = self._prepare_popup(
            camera_id, similarity, person_id, person_info, alert_time
        )

        # Step 4: 写入数据库
        db_result = self._write_to_database(
            camera_id, screenshot_path, similarity, alert_time,
            person_key, feature_vec, feature_type
        )

        # 组装预警记录
        alert_record = {
            'alert_time': alert_time.strftime('%Y-%m-%d %H:%M:%S'),
            'camera_id': camera_id,
            'similarity': similarity,
            'person_id': person_id,
            'person_info': person_info or {},
            'person_key': person_key,
            'screenshot_path': str(screenshot_path),
            'db_log_id': db_result.get('log_id', -1),
            'is_anomaly': True,
            'popup_info': alert_popup_info
        }

        # 添加到历史
        self.alert_history.append(alert_record)
        # 非阻塞入队: 队列满则丢最旧一条, 防止阻塞式 put() 在队列满时死锁检测循环
        try:
            self.alert_queue.put_nowait(alert_record)
        except queue.Full:
            try:
                self.alert_queue.get_nowait()
                self.alert_queue.put_nowait(alert_record)
            except queue.Empty:
                pass

        # 更新统计
        self.stats['total_alerts'] += 1
        self.stats['screenshots_saved'] += 1
        self.stats['db_writes'] += 1
        self.stats['last_alert_time'] = alert_time

        return alert_record

    def _save_screenshot(self, frame, camera_id, alert_time):
        """
        保存预警截图
        :return: 保存路径
        """
        timestamp = alert_time.strftime('%Y%m%d_%H%M%S_%f')[:16]
        filename = f"alert_{camera_id}_{timestamp}.jpg"
        filepath = self.output_dir / filename

        # 在截图上绘制预警信息
        annotated_frame = frame.copy()

        # 添加红色边框
        h, w = annotated_frame.shape[:2]
        cv2.rectangle(annotated_frame, (0, 0), (w-1, h-1), (0, 0, 255), 10)

        # 添加预警信息文本
        cv2.putText(annotated_frame, f"ALERT: {camera_id}", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        cv2.putText(annotated_frame,
                   f"Time: {alert_time.strftime('%Y-%m-%d %H:%M:%S')}",
                   (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 保存
        cv2.imwrite(str(filepath), annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])

        return filepath

    def _play_beep(self):
        """播放蜂鸣声"""
        current_time = time.time()

        # 冷却检查
        if current_time - self.last_beep_time < self.beep_cooldown:
            return

        self.last_beep_time = current_time

        try:
            # Windows 蜂鸣
            import winsound
            winsound.Beep(self.beep_frequency, self.beep_duration)
            self.stats['beeps_triggered'] += 1
        except ImportError:
            try:
                # 跨平台蜂鸣（ASCII BEL）
                print('\a', end='', flush=True)
                self.stats['beeps_triggered'] += 1
            except Exception:
                # 最后方案：打印提示
                print("[ALERT] 蜂鸣提示！")
                self.stats['beeps_triggered'] += 1

    def _prepare_popup(self, camera_id, similarity, person_id,
                       person_info, alert_time):
        """
        准备弹窗信息
        :return: 弹窗数据字典
        """
        person_info = person_info or {}
        return {
            'title': '⚠️ 异常人员预警',
            'message': f'摄像头 {camera_id} 检测到异常人员',
            'details': {
                'camera_id': camera_id,
                'alert_time': alert_time.strftime('%Y-%m-%d %H:%M:%S'),
                'similarity': f"{similarity:.4f}",
                'threshold': 0.85,
                'person_id': person_id,
                'person_name': person_info.get('name', '未知'),
                'person_room': person_info.get('room_num', '未知')
            },
            'color': '#ff4444',
            'is_anomaly': True,
            'action_required': 'security_notify'
        }

    def _write_to_database(self, camera_id, screenshot_path,
                            similarity, alert_time,
                            person_key=None, feature_vec=None, feature_type=None):
        """
        写入预警日志到 MySQL
        :return: {'log_id': 新记录ID, 'success': bool}
        """
        try:
            sys.path.insert(0, str(Path(__file__).parent))
            from db_mysql import HotelDatabase

            db = HotelDatabase(**self.db_config)
            if not db.test_connection():
                print("[WARN] 数据库连接失败，跳过写入")
                return {'log_id': -1, 'success': False}

            # 特征向量序列化为 JSON 数组存 TEXT
            feature_vec_json = None
            if feature_vec is not None:
                try:
                    feature_vec_json = json.dumps(np.asarray(feature_vec, dtype=np.float32).tolist())
                except Exception:
                    feature_vec_json = None

            log_id = db.insert_alert(
                camera_id=camera_id,
                screenshot_path=str(screenshot_path),
                similarity=similarity,
                handle_status=0,  # 未处理
                person_key=person_key,
                feature_vec=feature_vec_json,
                embedding_type=feature_type
            )
            db.close()

            return {'log_id': log_id, 'success': True}

        except ImportError:
            print("[WARN] 无法导入 db_mysql，跳过数据库写入")
            return {'log_id': -1, 'success': False}
        except Exception as e:
            print(f"[WARN] 数据库写入异常: {e}")
            return {'log_id': -1, 'success': False}

    @staticmethod
    def _normalize(vec):
        """L2 归一化特征向量; 输入 None 返回 None"""
        if vec is None:
            return None
        v = np.asarray(vec, dtype=np.float32).reshape(-1)
        norm = np.linalg.norm(v)
        if norm > 0:
            v = v / (norm + 1e-8)
        return v

    def _match_or_create(self, vec, field, threshold):
        """
        在指定特征维度 (field='face_vec' 或 'reid_vec') 上找最佳匹配簇;
        命中(余弦>=阈值)则复用其 key, 否则新建簇并存入该维度向量。
        """
        best_key, best_sim = None, -1.0
        for key, c in self._clusters.items():
            cv = c.get(field)
            if cv is None:
                continue
            sim = float(np.dot(cv, vec))
            if sim > best_sim:
                best_sim, best_key = sim, key

        if best_key is not None and best_sim >= threshold:
            return best_key

        self._cluster_seq += 1
        new_key = f"anom_{int(time.time() * 1000)}_{self._cluster_seq}"
        self._clusters[new_key] = {'face_vec': None, 'reid_vec': None}
        self._clusters[new_key][field] = vec
        return new_key

    def _assign_person_key(self, face_vec, reid_vec):
        """
        给异常人员分配/复用 person_key (人脸优先 + ReID 兜底, 支持 face↔reid 跨类型合并)。
        - 人脸簇和 ReID 簇分别用各自阈值聚类;
        - 同一次检测同时命中人脸簇和 ReID 簇 → 判定为同一人, 合并 (人脸簇为规范 key)。
        - 只有脸/只有全身 → 归入对应簇 (若带了另一路特征则补充存进该簇, 供后续合并)。
        - 两路特征都无 → 每次独立 key。
        """
        face_vec = self._normalize(face_vec)
        reid_vec = self._normalize(reid_vec)

        face_key = (self._match_or_create(face_vec, 'face_vec', self.FACE_CLUSTER_THRESHOLD)
                    if face_vec is not None else None)
        reid_key = (self._match_or_create(reid_vec, 'reid_vec', self.REID_CLUSTER_THRESHOLD)
                    if reid_vec is not None else None)

        if face_key is None and reid_key is None:
            self._cluster_seq += 1
            return f"anom_nofeat_{int(time.time() * 1000)}_{self._cluster_seq}"

        if face_key is None:
            return reid_key

        if reid_key is None:
            # 只有脸命中 (没带 reid 或 reid 未命中): 若带了 reid 特征则补充进该簇
            if reid_vec is not None and self._clusters[face_key].get('reid_vec') is None:
                self._clusters[face_key]['reid_vec'] = reid_vec
            return face_key

        if face_key == reid_key:
            return face_key

        # 脸和全身分别命中不同簇 → 同一人, 合并 (人脸簇为规范 key)
        return self._merge(reid_key, face_key)

    def _merge(self, from_key, to_key):
        """把 from_key 簇合并进 to_key 簇 (特征取并集), 并同步更新 DB 里 from_key 的旧预警记录"""
        f = self._clusters.get(from_key)
        t = self._clusters.get(to_key)
        if f and t:
            for field in ('face_vec', 'reid_vec'):
                if t.get(field) is None and f.get(field) is not None:
                    t[field] = f[field]
        self._clusters.pop(from_key, None)
        self._rekey_db(from_key, to_key)
        return to_key

    def _rekey_db(self, old_key, new_key):
        """把 DB 中 old_key 的预警记录改归 new_key (跨类型合并到同一卡片)"""
        try:
            sys.path.insert(0, str(Path(__file__).parent))
            from db_mysql import HotelDatabase
            db = HotelDatabase(**self.db_config)
            if db.test_connection():
                n = db.update_alert_person_key(old_key, new_key)
                db.close()
                if n:
                    print(f"[INFO] 跨类型合并: {old_key} -> {new_key} (更新 {n} 条)")
        except Exception as e:
            print(f"[WARN] 跨类型合并写库失败: {e}")

    def _rebuild_clusters(self):
        """从已有预警记录重建按人聚类 (重启后同一人仍归入同一 person_key)。
        同一 person_key 可同时拥有 face_vec 与 reid_vec (来自之前跨类型合并写库的两类记录)。"""
        try:
            sys.path.insert(0, str(Path(__file__).parent))
            from db_mysql import HotelDatabase

            db = HotelDatabase(**self.db_config)
            if not db.test_connection():
                db.close()
                return
            alerts = db.get_all_alerts(limit=100000)
            db.close()

            max_seq = 0
            for a in alerts:
                key = a.get('person_key')
                if not key:
                    continue
                try:
                    vec = np.array(json.loads(a['feature_vec']), dtype=np.float32).reshape(-1)
                except (json.JSONDecodeError, ValueError, TypeError):
                    continue
                vec = self._normalize(vec)
                if vec is None:
                    continue
                etype = a.get('embedding_type') or 'reid'
                field = 'face_vec' if etype == 'face' else 'reid_vec'
                cl = self._clusters.setdefault(key, {'face_vec': None, 'reid_vec': None})
                if cl[field] is None:
                    cl[field] = vec
                try:
                    max_seq = max(max_seq, int(key.rsplit('_', 1)[-1]))
                except (ValueError, IndexError):
                    pass

            self._cluster_seq = max_seq
            print(f"[INFO] 从预警记录重建 {len(self._clusters)} 个异常人员簇")
        except Exception as e:
            print(f"[WARN] 重建异常人员簇失败: {e}")

    def get_next_alert(self, timeout=0.1):
        """获取队列中下一个预警"""
        try:
            return self.alert_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def get_stats(self):
        """获取统计信息"""
        return self.stats.copy()

    def get_recent_alerts(self, count=10):
        """获取最近预警"""
        return self.alert_history[-count:]

    def acknowledge_alert(self, log_id, status=1):
        """确认/处置预警"""
        try:
            sys.path.insert(0, str(Path(__file__).parent))
            from db_mysql import HotelDatabase

            db = HotelDatabase(**self.db_config)
            success = db.update_alert_status(log_id, status)
            db.close()
            return success
        except Exception as e:
            print(f"[WARN] 预警处置异常: {e}")
            return False


class AnomalyDetector:
    """
    异常检测器（集成 ReID + 预警）
    - YOLO 检测行人 → ReID 匹配 → 异常判定 → 预警触发
    """

    def __init__(self, video_stream, reid_pipeline, alert_manager):
        """
        :param video_stream: MultiCameraManager 实例
        :param reid_pipeline: ReidMatchingPipeline 实例
        :param alert_manager: AlertManager 实例
        """
        self.video_stream = video_stream
        self.reid_pipeline = reid_pipeline
        self.alert_manager = alert_manager

        # 检测状态
        self.is_running = False
        self.detection_thread = None

        # 最近检测结果
        self.last_detections = {}

        # 状态变化检测（防抖）
        self.alert_cooldown_map = {}  # camera_id -> last_alert_time

        # 统计
        self.stats = {
            'total_persons_detected': 0,
            'matched_persons': 0,
            'unmatched_persons': 0,
            'alerts_triggered': 0,
            'avg_processing_time_ms': 0.0
        }

    def start(self):
        """启动检测循环"""
        self.is_running = True
        self.detection_thread = threading.Thread(
            target=self._detection_loop, daemon=True
        )
        self.detection_thread.start()
        print(f"[INFO] 异常检测已启动")

    def stop(self):
        """停止检测"""
        self.is_running = False
        if self.detection_thread:
            self.detection_thread.join(timeout=2)
        print(f"[INFO] 异常检测已停止")

    def _detection_loop(self):
        """检测主循环"""
        try:
            while self.is_running:
                result = self.video_stream.get_result(timeout=0.1)
                if result is None:
                    continue

                camera_id = result['camera_id']
                frame = result['frame']
                detections = result['detections']

                # 处理每个检测到的行人
                for det in detections:
                    processing_start = time.time()

                    # 获取 ROI
                    roi = det.get('roi')
                    if roi is not None and roi.size > 0:
                        # ReID 匹配
                        match_result = self.reid_pipeline.process(roi)

                        # 判定异常
                        if match_result['is_anomaly']:
                            self._handle_anomaly(
                                camera_id, frame,
                                match_result['similarity'],
                                match_result.get('person_id', -1),
                                match_result.get('person_info', {})
                            )
                        else:
                            self._handle_matched(
                                camera_id,
                                match_result.get('person_info', {})
                            )

                        # 更新统计
                        self.stats['total_persons_detected'] += 1
                        if match_result['is_matched']:
                            self.stats['matched_persons'] += 1
                        else:
                            self.stats['unmatched_persons'] += 1

                        # 更新平均处理时间
                        proc_time = (time.time() - processing_start) * 1000
                        total = self.stats['total_persons_detected']
                        prev_avg = self.stats['avg_processing_time_ms']
                        self.stats['avg_processing_time_ms'] = prev_avg + (proc_time - prev_avg) / total

                # 存储最近结果
                self.last_detections[camera_id] = {
                    'timestamp': time.time(),
                    'detections': detections,
                    'frame': frame.copy()
                }

        except Exception as e:
            print(f"[ERROR] 检测循环异常: {e}")

    def _handle_anomaly(self, camera_id, frame, similarity,
                        person_id, person_info):
        """处理异常人员"""
        current_time = time.time()

        # 预警冷却检查（同一摄像头短时间内不重复触发）
        last_alert = self.alert_cooldown_map.get(camera_id, 0)
        if current_time - last_alert < 10.0:  # 10秒冷却
            return

        # 触发预警
        alert_record = self.alert_manager.trigger_alert(
            camera_id=camera_id,
            frame=frame,
            similarity=similarity,
            person_id=person_id,
            person_info=person_info
        )

        self.alert_cooldown_map[camera_id] = current_time
        self.stats['alerts_triggered'] += 1

        print(f"[ALERT] {camera_id} | 相似度: {similarity:.4f} | "
              f"预警ID: {alert_record['db_log_id']}")

    def _handle_matched(self, camera_id, person_info):
        """处理已登记人员"""
        name = person_info.get('name', '未知')
        room = person_info.get('room_num', '未知')

        # 记录但不预警
        pass  # 正常通过，无特殊动作

    def get_current_status(self):
        """获取当前状态"""
        return {
            'is_running': self.is_running,
            'stats': self.stats.copy(),
            'last_detections': list(self.last_detections.keys()),
            'alerts_in_queue': self.alert_manager.alert_queue.qsize()
        }


# ============================================================
# 独立调试入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="异常预警管理器调试工具")
    parser.add_argument("--test", action="store_true",
                        help="运行模拟预警测试")
    parser.add_argument("--count", type=int, default=5,
                        help="模拟预警次数")
    parser.add_argument("--interval", type=float, default=1.0,
                        help="预警间隔秒数")
    parser.add_argument("--output", type=str,
                        default="output/alert_screenshots",
                        help="截图输出目录")

    args = parser.parse_args()

    print("=" * 60)
    print("异常预警管理器调试工具")
    print("=" * 60)

    if args.test:
        # 初始化管理器
        manager = AlertManager(output_dir=args.output)

        # 模拟预警测试
        print(f"\n[INFO] 开始模拟预警测试 ({args.count} 次, 间隔 {args.interval}s)...\n")

        for i in range(args.count):
            # 生成模拟帧
            frame = np.random.randint(100, 200, (480, 640, 3), dtype=np.uint8)

            # 模拟检测框
            cv2.rectangle(frame, (100, 100), (200, 400), (0, 255, 0), 2)
            cv2.putText(frame, f"Simulated Frame #{i+1}", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # 触发预警（使用低相似度模拟异常）
            similarity = 0.5 + 0.3 * np.random.random()  # 0.5 ~ 0.8

            camera_id = f"CAM_{(i % 4) + 1:03d}"

            alert_record = manager.trigger_alert(
                camera_id=camera_id,
                frame=frame,
                similarity=similarity,
                person_id=-1,
                person_info={}
            )

            print(f"  [{i+1}/{args.count}] {alert_record['alert_time']} | "
                  f"摄像头: {camera_id} | 相似度: {similarity:.4f} | "
                  f"截图: {Path(alert_record['screenshot_path']).name} | "
                  f"DB ID: {alert_record['db_log_id']}")

            time.sleep(args.interval)

        # 输出统计
        print("\n" + "=" * 60)
        print("预警测试统计结果")
        print("=" * 60)

        stats = manager.get_stats()
        for key, value in stats.items():
            if key != 'last_alert_time':
                print(f"  {key}: {value}")

        print(f"\n  最近预警记录:")
        for alert in manager.get_recent_alerts(5):
            print(f"    [{alert['alert_time']}] {alert['camera_id']} | "
                  f"相似度: {alert['similarity']:.4f} | "
                  f"截图: {Path(alert['screenshot_path']).name}")

        print(f"\n  截图保存位置: {manager.output_dir}")
        screenshots = list(manager.output_dir.glob('*.jpg'))
        print(f"  已保存截图数量: {len(screenshots)}")

        # 验证三项动作
        print("\n" + "=" * 60)
        print("三项动作验证")
        print("=" * 60)

        # 1. 截图保存
        if screenshots:
            print(f"  ✓ 截图保存: {len(screenshots)} 张")
        else:
            print(f"  ✗ 截图保存: 失败")

        # 2. 蜂鸣触发
        print(f"  {'✓' if stats['beeps_triggered'] > 0 else '✗'} "
              f"蜂鸣提示: 触发 {stats['beeps_triggered']} 次")

        # 3. 数据库写入
        if stats['db_writes'] > 0:
            print(f"  ✓ 数据库写入: 成功 {stats['db_writes']} 条")
        else:
            print(f"  ✗ 数据库写入: 可能未连接 MySQL")

    else:
        print("[INFO] 使用 --test 参数运行模拟预警测试")


if __name__ == '__main__':
    main()