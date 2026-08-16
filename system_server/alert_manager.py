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
    # 实测 (CAM_001 单人 30s 连续 49 个 ReID 特征): 两两余弦 中位数 0.23, 75 分位 0.51, 90 分位 0.85。
    # 同一人的 ReID 特征呈双峰: 清晰大目标 0.5~0.9, 小目标/遮挡退化后 <0.3。0.55 会把同人拆散,
    # 0.35 可合并同人 (退化样本交空间-时序兜底), 又高于不同人的 <0.2 区。人脸 512 维更具判别性, 更低。
    FACE_CLUSTER_THRESHOLD = 0.4    # 人脸 512 维
    REID_CLUSTER_THRESHOLD = 0.35   # ReID 2048 维
    CLUSTER_MAX_EXEMPLARS = 5       # 每个簇每维度保留的历史样例数 (多样例取最大余弦, 抗姿态/视角多模态)
    TRACK_IOU_THRESHOLD = 0.3       # 空间-时序兜底: bbox IoU 下限 (同一人连续出现)
    TRACK_TTL = 3.0                 # 活跃 track 存活秒数 (超过视为可能换了人)

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

        # 按人聚类状态: person_key -> {'face_vec': [512维人脸向量...], 'reid_vec': [2048维全身向量...]}
        self._clusters = {}
        self._cluster_seq = 0  # 自增序号, 保证新 key 不重复

        # 空间-时序 track: person_key -> {'bbox': [x1,y1,x2,y2], 'ts': 最后出现时间戳}
        # 特征偶发退化 (小目标/遮挡) 时, 靠 bbox IoU 复用"同一人连续出现"的 key, 防止拆散
        self._tracks = {}

        # 从已有预警记录重建聚类 (重启后同一人仍归入同一 person_key)
        self._rebuild_clusters()

        print(f"[INFO] 预警管理器初始化完成")
        print(f"[INFO] 截图保存目录: {self.output_dir}")

    def trigger_alert(self, camera_id, frame, similarity,
                      person_id=-1, person_info=None,
                      feature_vec=None, feature_type=None,
                      face_feature_vec=None, reid_feature_vec=None,
                      bbox=None):
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
        :param bbox: 行人框 [x1,y1,x2,y2], 用于把截图裁剪到该人 (缺省则整帧)
        :return: 预警记录字典
        """
        # 按人聚类: 分配/复用 person_key (人脸优先 + ReID 兜底, 且 face↔reid 跨类型合并)
        face_vec = (face_feature_vec if face_feature_vec is not None
                    else (feature_vec if feature_type == 'face' else None))
        reid_vec = (reid_feature_vec if reid_feature_vec is not None
                    else (feature_vec if feature_type == 'reid' else None))
        person_key = self._assign_person_key(face_vec, reid_vec, bbox=bbox)

        # 预警冷却: 按 (摄像头, 人) 维度, 同人冷却期内不重复触发, 不同人各自立即触发
        now = time.time()
        cooldown_key = (camera_id, person_key)
        if now - self._last_alert_time.get(cooldown_key, 0) < self.alert_cooldown:
            return None
        self._last_alert_time[cooldown_key] = now

        alert_time = datetime.now()

        # Step 1: 保存截图 (裁剪到异常人员个体, 对应其身份)
        screenshot_path = self._save_screenshot(frame, camera_id, alert_time, bbox=bbox)

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

    def _save_screenshot(self, frame, camera_id, alert_time, bbox=None):
        """
        保存预警截图
        :param bbox: 行人框 [x1,y1,x2,y2]; 提供时裁剪到该人 (带边距), 否则整帧
        :return: 保存路径
        """
        timestamp = alert_time.strftime('%Y%m%d_%H%M%S_%f')[:16]
        filename = f"alert_{camera_id}_{timestamp}.jpg"
        filepath = self.output_dir / filename

        # 裁剪到异常人员个体 (带边距), 让截图对应到"这个人"的身份, 而非整个监控画面
        if bbox is not None and frame is not None:
            annotated_frame = self._crop_person(frame, bbox)
        else:
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

    @staticmethod
    def _crop_person(frame, bbox, margin=0.25):
        """按行人框裁剪个体图, 四周各留 margin 比例边距, 坐标钳制到帧内"""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        pad_x = int(bw * margin)
        pad_y = int(bh * margin)
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)
        if x2 <= x1 or y2 <= y1:
            return frame.copy()
        return frame[y1:y2, x1:x2].copy()

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

    def _match(self, vec, field, threshold):
        """
        在指定特征维度 (field='face_vec' 或 'reid_vec') 上找最佳匹配簇 (不创建)。
        与簇内全部历史样例取「最大」余弦 (对姿态/视角多模态更鲁棒);
        命中返回 key, 未命中返回 None。
        """
        if vec is None:
            return None
        best_key, best_sim = None, -1.0
        for key, c in self._clusters.items():
            exemplars = c.get(field)
            if not exemplars:
                continue
            sim = float(np.max(np.dot(np.asarray(exemplars, dtype=np.float32), vec)))
            if sim > best_sim:
                best_sim, best_key = sim, key
        if best_key is not None and best_sim >= threshold:
            return best_key
        return None

    def _append_feature(self, key, field, vec):
        """把 vec 并入簇的指定维度样例 (保留最近 CLUSTER_MAX_EXEMPLARS 个, 防止无界膨胀)"""
        if key is None or vec is None:
            return
        cl = self._clusters.setdefault(key, {'face_vec': [], 'reid_vec': []})
        lst = cl.setdefault(field, [])
        lst.append(vec)
        if len(lst) > self.CLUSTER_MAX_EXEMPLARS:
            lst.pop(0)

    def _new_cluster(self, face_vec=None, reid_vec=None):
        """新建簇并存入初始特征样例"""
        self._cluster_seq += 1
        new_key = f"anom_{int(time.time() * 1000)}_{self._cluster_seq}"
        self._clusters[new_key] = {'face_vec': [], 'reid_vec': []}
        self._append_feature(new_key, 'face_vec', face_vec)
        self._append_feature(new_key, 'reid_vec', reid_vec)
        return new_key

    @staticmethod
    def _bbox_iou(a, b):
        """计算两个 [x1,y1,x2,y2] 框的 IoU"""
        if a is None or b is None:
            return 0.0
        ax1, ay1, ax2, ay2 = [float(v) for v in a]
        bx1, by1, bx2, by2 = [float(v) for v in b]
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        area_a = max(1.0, (ax2 - ax1) * (ay2 - ay1))
        area_b = max(1.0, (bx2 - bx1) * (by2 - by1))
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def _match_track(self, bbox):
        """空间-时序兜底: 找与 bbox 空间重叠且 TRACK_TTL 内出现过的活跃 track, 复用其 key"""
        if bbox is None:
            return None
        now = time.time()
        # 清理过期 track, 防止长期运行无界膨胀
        stale = [k for k, tr in self._tracks.items() if now - tr['ts'] > self.TRACK_TTL]
        for k in stale:
            self._tracks.pop(k, None)

        best_key, best_iou = None, self.TRACK_IOU_THRESHOLD
        for key, tr in self._tracks.items():
            iou = self._bbox_iou(bbox, tr['bbox'])
            if iou > best_iou:
                best_iou, best_key = iou, key
        return best_key

    def _update_track(self, key, bbox):
        """更新 key 对应的活跃 track (bbox + 时间戳)"""
        if key is None or bbox is None:
            return
        self._tracks[key] = {'bbox': [float(v) for v in bbox], 'ts': time.time()}

    def _resolve_keys(self, face_key, reid_key, face_vec, reid_vec):
        """
        融合人脸/ReID 命中结果:
        - 都未命中 → None
        - 仅人脸 / 仅 ReID → 对应 key
        - 都命中不同 key → 同一人, 合并 (人脸簇为规范 key)
        命中后把本次多模态特征并入簇, 更新历史样例。
        """
        if face_key is not None and reid_key is not None and face_key != reid_key:
            self._merge(reid_key, face_key)

        key = face_key if face_key is not None else reid_key
        if key is None:
            return None
        self._append_feature(key, 'face_vec', face_vec)
        self._append_feature(key, 'reid_vec', reid_vec)
        return key

    def _assign_person_key(self, face_vec, reid_vec, bbox=None):
        """
        给异常人员分配/复用 person_key (空间-时序优先, 人脸+ReID 兜底)。
        三级策略:
          1) 空间-时序优先: bbox 与活跃 track 重叠 (同一人连续出现) → 直接复用其 key,
             即使特征偶发退化 (小目标/遮挡/姿态变化) 也不拆散;
          2) 特征聚类: 无时序重叠 (新人/离散出现) → 人脸/ReID 按阈值匹配或新建簇;
          3) 全未命中 → 新建簇。
        同一条 track 上的人脸与全身特征会自然并入同一簇, 无需额外跨类型合并。
        """
        face_vec = self._normalize(face_vec)
        reid_vec = self._normalize(reid_vec)

        # 1) 空间-时序优先 (同一人连续出现, 靠 bbox IoU 维持身份连续性)
        track_key = self._match_track(bbox)
        if track_key is not None:
            self._append_feature(track_key, 'face_vec', face_vec)
            self._append_feature(track_key, 'reid_vec', reid_vec)
            self._update_track(track_key, bbox)
            return track_key

        # 2) 特征聚类 (无时序重叠 → 新人或离散出现)
        face_key = self._match(face_vec, 'face_vec', self.FACE_CLUSTER_THRESHOLD)
        reid_key = self._match(reid_vec, 'reid_vec', self.REID_CLUSTER_THRESHOLD)
        key = self._resolve_keys(face_key, reid_key, face_vec, reid_vec)
        if key is not None:
            self._update_track(key, bbox)
            return key

        # 3) 全未命中 → 新建簇
        key = self._new_cluster(face_vec, reid_vec)
        self._update_track(key, bbox)
        return key

    def _merge(self, from_key, to_key):
        """把 from_key 簇合并进 to_key 簇 (特征样例取并集), 并同步更新 DB 里 from_key 的旧预警记录"""
        f = self._clusters.get(from_key)
        t = self._clusters.get(to_key)
        if f and t:
            for field in ('face_vec', 'reid_vec'):
                if f.get(field):
                    merged = (t.get(field) or []) + f.get(field)
                    t[field] = merged[-self.CLUSTER_MAX_EXEMPLARS:]
        self._clusters.pop(from_key, None)
        self._tracks.pop(from_key, None)
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
                cl = self._clusters.setdefault(key, {'face_vec': [], 'reid_vec': []})
                if len(cl[field]) < self.CLUSTER_MAX_EXEMPLARS:
                    cl[field].append(vec)
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