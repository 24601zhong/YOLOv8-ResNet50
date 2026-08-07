# -*- coding: -*-8 -*-
"""
Flask MJPEG 视频流服务 mjpeg_server.py
======================================
功能：
  - MJPEG 协议多路视频推流
  - 画面叠加绿框（已登记）/红框（异常人员）
  - 侧边栏展示系统 FPS、在线人数、预警数
  - 支持前后端 WebSocket/SSE 实时通信
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
import torch
from flask import Flask, Response, render_template, jsonify, request
from flask_cors import CORS

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from system_server.video_stream import YOLOPersonDetector, MultiCameraManager
from system_server.reid_match import ReidMatchingPipeline
from system_server.alert_manager import AlertManager, AnomalyDetector


class MJPEGStreamServer:
    """
    MJPEG 多路视频流服务器
    - 为每个摄像头创建独立的 MJPEG 流
    - 实时叠加检测框和身份标签
    """

    MJPEG_CONTENT_TYPE = 'multipart/x-mixed-replace; boundary=frame'

    def __init__(self, host='0.0.0.0', port=5001):
        self.host = host
        self.port = port

        # Flask 应用
        self.app = Flask(__name__)
        CORS(self.app)

        # 检测组件
        self.detector = None
        self.reid_pipeline = None
        self.alert_manager = None
        self.anomaly_detector = None

        # 摄像头帧缓冲区（用于 MJPEG 推流）
        self.frame_buffers = {}  # camera_id -> {'frame', 'timestamp', 'detections'}
        self.frame_lock = threading.Lock()

        # 全局统计
        self.system_stats = {
            'fps': 0.0,
            'online_cameras': 0,
            'total_detections': 0,
            'matched_persons': 0,
            'alerts_today': 0,
            'start_time': time.time()
        }

        self._register_routes()

    def _register_routes(self):
        """注册 Flask 路由"""
        # MJPEG 视频流
        self.app.add_url_rule(
            '/video_feed/<camera_id>', 'video_feed',
            self._video_feed_endpoint
        )

        # 状态 API
        self.app.add_url_rule(
            '/api/system_status', 'system_status',
            self._system_status_endpoint
        )

        # 摄像头列表
        self.app.add_url_rule(
            '/api/cameras', 'cameras_list',
            self._cameras_list_endpoint
        )

        # 健康检查
        self.app.add_url_rule(
            '/health', 'health_check',
            self._health_check_endpoint
        )

    def _video_feed_endpoint(self, camera_id):
        """
        MJPEG 视频流端点
        浏览器访问 /video_feed/CAM_001 获取实时流
        """
        def generate():
            boundary = b'--frame\r\n'
            content_type = b'Content-Type: image/jpeg\r\n\r\n'

            while True:
                buffer_data = self.frame_buffers.get(camera_id)

                if buffer_data is not None:
                    frame = buffer_data.get('frame')
                    if frame is not None:
                        # 编码为 JPEG
                        _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])

                        # MJPEG 格式
                        yield boundary + content_type + jpeg.tobytes() + b'\r\n'

                # 避免 CPU 过载
                time.sleep(0.01)

        return Response(generate(), content_type=self.MJPEG_CONTENT_TYPE)

    def _system_status_endpoint(self):
        """系统状态 API"""
        return jsonify({
            'status': 'running',
            'stats': self.system_stats,
            'cameras': list(self.frame_buffers.keys()),
            'anomaly_detector': self.anomaly_detector.get_current_status() if self.anomaly_detector else {}
        })

    def _cameras_list_endpoint(self):
        """摄像头列表 API"""
        cameras = []
        for cam_id in self.frame_buffers:
            buffer = self.frame_buffers[cam_id]
            cameras.append({
                'id': cam_id,
                'has_frame': buffer.get('frame') is not None,
                'timestamp': buffer.get('timestamp')
            })
        return jsonify(cameras)

    def _health_check_endpoint(self):
        """健康检查"""
        return jsonify({
            'status': 'healthy',
            'uptime': time.time() - self.system_stats['start_time']
        })

    def initialize_components(self, yolo_model_path=None, reid_model_path=None,
                               db_config=None):
        """初始化检测组件"""
        print("[INFO] 初始化检测组件...")

        # 1. YOLO 检测器
        self.detector = YOLOPersonDetector(model_path=yolo_model_path)

        # 2. ReID 流水线
        self.reid_pipeline = ReidMatchingPipeline(
            model_path=reid_model_path,
            db_config=db_config
        )

        # 3. 预警管理器
        self.alert_manager = AlertManager(db_config=db_config)

        # 4. 异常检测器
        self.anomaly_detector = AnomalyDetector(
            video_stream=None,  # 由 VideoStreamBridge 桥接
            reid_pipeline=self.reid_pipeline,
            alert_manager=self.alert_manager
        )

        print("[INFO] 检测组件初始化完成")

    def update_frame(self, camera_id, frame, detections=None, person_results=None):
        """
        更新摄像头帧缓冲区（供 MJPEG 推流使用）
        由外部检测循环调用
        """
        # 绘制检测框
        annotated_frame = frame.copy()

        if detections:
            for det in detections:
                bbox = det.get('bbox', [0, 0, 0, 0])
                x1, y1, x2, y2 = [int(v) for v in bbox]
                confidence = det.get('confidence', 0)

                # 绘制半透明背景
                overlay = annotated_frame.copy()

                # 绿色框（已登记）或红色框（异常）
                is_matched = det.get('is_matched', False)
                is_anomaly = det.get('is_anomaly', not is_matched)

                if is_anomaly:
                    color = (0, 0, 255)  # 红色
                    label_text = f"ALERT: {confidence:.2f}"
                else:
                    color = (0, 255, 0)  # 绿色
                    person_name = det.get('person_name', 'Guest')
                    person_room = det.get('person_room', '')
                    label_text = f"{person_name} Room:{person_room}"

                # 绘制边框
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 3)

                # 绘制标签背景
                (text_width, text_height), _ = cv2.getTextSize(
                    label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                cv2.rectangle(overlay,
                             (x1, y1 - text_height - 10),
                             (x1 + text_width + 10, y1),
                             color, -1)

                # 绘制标签文字
                cv2.putText(overlay, label_text,
                           (x1 + 5, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # 合并透明度
                cv2.addWeighted(overlay, 0.7, annotated_frame, 0.3, 0, annotated_frame)

        # 添加时间戳和摄像头信息
        info_y = annotated_frame.shape[0] - 10
        timestamp = datetime.now().strftime('%H:%M:%S')
        info_text = f"{camera_id} | {timestamp}"
        cv2.putText(annotated_frame, info_text,
                   (10, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 更新缓冲区
        with self.frame_lock:
            self.frame_buffers[camera_id] = {
                'frame': annotated_frame,
                'timestamp': time.time(),
                'detections': detections or []
            }

        # 更新系统统计
        self.system_stats['online_cameras'] = len(self.frame_buffers)

    def update_stats(self, stats_update):
        """更新系统统计"""
        self.system_stats.update(stats_update)

    def start(self):
        """启动 MJPEG 服务器"""
        print(f"\n[INFO] MJPEG 视频流服务器启动")
        print(f"[INFO] 访问地址: http://{self.host}:{self.port}")
        print(f"[INFO] 视频流: http://{self.host}:{self.port}/video_feed/<camera_id>\n")

        self.app.run(host=self.host, port=self.port, debug=False, threaded=True)


# ============================================================
# 桥接类：连接视频流检测与 MJPEG 推流
# ============================================================

class DetectionStreamBridge:
    """
    检测-推流桥接器
    - 将 MultiCameraManager 的检测结果转换为 MJPEG 帧
    """

    def __init__(self, camera_manager, mjpeg_server, reid_pipeline,
                 match_threshold=0.85):
        self.camera_manager = camera_manager
        self.mjpeg_server = mjpeg_server
        self.reid_pipeline = reid_pipeline
        self.match_threshold = match_threshold

        # 处理统计
        self.processed_count = 0
        self.matched_count = 0
        self.alert_count = 0

        # 系统统计
        self.fps_values = []

    def start_processing(self):
        """启动处理循环"""
        print("[INFO] 启动检测-推流桥接处理...")

        while self.camera_manager.running:
            result = self.camera_manager.get_result(timeout=0.1)
            if result is None:
                continue

            camera_id = result['camera_id']
            frame = result['frame']
            raw_detections = result['detections']

            processed_detections = []

            # 对每个检测行人进行 ReID 匹配
            for det in raw_detections:
                roi = det.get('roi')
                if roi is not None and roi.size > 0:
                    # ReID 匹配
                    match_result = self.reid_pipeline.process(roi)

                    det['is_matched'] = match_result['is_matched']
                    det['is_anomaly'] = match_result['is_anomaly']
                    det['similarity'] = match_result['similarity']
                    det['person_name'] = match_result['person_info'].get('name', '未知')
                    det['person_room'] = match_result['person_info'].get('room_num', '')

                    # 异常触发预警
                    if match_result['is_anomaly']:
                        self.alert_count += 1
                        self.mjpeg_server.alert_manager.trigger_alert(
                            camera_id=camera_id,
                            frame=frame,
                            similarity=match_result['similarity'],
                            person_id=match_result.get('person_id', -1),
                            person_info=match_result['person_info']
                        )
                    else:
                        self.matched_count += 1

                processed_detections.append(det)

            # 更新 MJPEG 帧
            self.mjpeg_server.update_frame(camera_id, frame, processed_detections)

            # 更新统计
            self.processed_count += 1

            if self.processed_count % 30 == 0:
                fps = self.processed_count / max(time.time() - self.mjpeg_server.system_stats['start_time'], 1)
                self.fps_values.append(fps)
                avg_fps = np.mean(self.fps_values[-30:]) if self.fps_values else 0
                self.mjpeg_server.update_stats({
                    'fps': avg_fps,
                    'total_detections': self.processed_count,
                    'matched_persons': self.matched_count,
                    'alerts_today': self.alert_count
                })

    def get_stats(self):
        return {
            'processed': self.processed_count,
            'matched': self.matched_count,
            'alerts': self.alert_count,
            'avg_fps': np.mean(self.fps_values[-30:]) if self.fps_values else 0
        }


# ============================================================
# 独立测试入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="MJPEG 视频流服务器")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5001)
    parser.add_argument("--video", type=str,
                        default="test_video/hotel_raw/camera_01.mp4",
                        help="测试视频路径")
    parser.add_argument("--yolo_model", type=str, default=None)
    parser.add_argument("--reid_model", type=str, default=None)
    parser.add_argument("--skip_frame", type=int, default=2)

    args = parser.parse_args()

    print("=" * 60)
    print("MJPEG 多路视频流服务器")
    print("=" * 60)

    # 1. 创建 MJPEG 服务器
    mjpeg_server = MJPEGStreamServer(host=args.host, port=args.port)

    # 2. 创建检测器
    detector = YOLOPersonDetector(model_path=args.yolo_model)
    reid_pipeline = ReidMatchingPipeline(model_path=args.reid_model)
    alert_manager = AlertManager()

    # 3. 创建摄像头管理器
    camera_manager = MultiCameraManager(detector, skip_frame=args.skip_frame)

    # 添加测试视频
    video_path = Path(args.video)
    if video_path.exists():
        camera_manager.add_camera(str(video_path), "CAM_001")
    else:
        # 使用摄像头
        camera_manager.add_camera(0, "CAM_001")
        print("[WARN] 测试视频不存在，使用默认摄像头")

    # 4. 创建桥接器
    bridge = DetectionStreamBridge(
        camera_manager, mjpeg_server, reid_pipeline
    )

    # 5. 启动处理线程
    processing_thread = threading.Thread(target=bridge.start_processing, daemon=True)
    processing_thread.start()

    # 6. 启动摄像头读取
    read_thread = threading.Thread(
        target=camera_manager.start_detection_loop, daemon=True
    )
    read_thread.start()

    # 7. 启动 Flask 服务器（阻塞）
    print(f"\n[INFO] 服务器启动中...")
    print(f"[INFO] 浏览器访问: http://localhost:{args.port}")
    print(f"[INFO] 视频流地址: http://localhost:{args.port}/video_feed/CAM_001\n")

    try:
        mjpeg_server.start()
    except KeyboardInterrupt:
        print("\n[INFO] 服务器停止")
        camera_manager.stop()


if __name__ == '__main__':
    main()