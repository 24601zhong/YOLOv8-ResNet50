# -*- coding: utf-8 -*-
"""
多路视频流解析 + YOLO 行人检测模块 video_stream.py
==================================================
功能：
  1. OpenCV 多线程读取 RTSP 网络摄像头流 / 本地测试视频
  2. 跳帧策略：每 2 帧送入 YOLO 推理（平衡算力与实时性）
  3. 检测后裁剪行人 ROI，统一缩放至 256×128
  4. 独立调试标准：视频画面正常、检测框稳定、无漏检、无崩溃
"""

import os
import sys
import time
import threading
import queue
import argparse
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

import cv2
import torch


# ============================================================
# 摄像头读取线程（规避 IO 阻塞）
# ============================================================

class CameraStreamReader:
    """
    多线程视频流读取器
    支持 RTSP 网络摄像头和本地视频文件
    """

    def __init__(self, source, camera_id, queue_size=2, skip_frame=2):
        """
        :param source: 视频源（RTSP地址或本地文件路径）
        :param camera_id: 摄像头编号
        :param queue_size: 帧队列大小
        :param skip_frame: 跳帧间隔（每N帧取1帧送入推理）
        """
        self.source = source
        self.camera_id = camera_id
        self.skip_frame = skip_frame
        self.frame_queue = queue.Queue(maxsize=queue_size)

        self.cap = None
        self.thread = None
        self.running = False
        self.frame_count = 0
        self.dropped_frames = 0

        # 判断视频源类型
        source_str = str(source) if not isinstance(source, str) else source
        is_rtsp = source_str.startswith('rtsp://')
        is_file = not is_rtsp and isinstance(source, str)

        # 统计信息
        self.stats = {
            'camera_id': camera_id,
            'source': source,
            'total_frames_read': 0,
            'frames_sent_to_model': 0,
            'dropped_frames': 0,
            'fps': 0.0,
            'resolution': (0, 0),
            'is_rtsp': is_rtsp,
            'is_file': is_file,
            'is_camera': isinstance(source, int)
        }

    def start(self):
        """启动读取线程"""
        self.cap = cv2.VideoCapture(self.source)

        if not self.cap.isOpened():
            print(f"[ERROR] 摄像头 {self.camera_id} 无法打开: {self.source}")
            return False

        self.stats['resolution'] = (
            int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        )

        self.running = True
        self.thread = threading.Thread(target=self._read_frames, daemon=True)
        self.thread.start()

        print(f"[INFO] 摄像头 {self.camera_id} 已启动: {self.source} "
              f"(分辨率: {self.stats['resolution']})")
        return True

    def _read_frames(self):
        """帧读取循环（独立线程）"""
        last_frame_time = time.time()
        fps_counter = 0

        while self.running:
            ret, frame = self.cap.read()

            if not ret:
                if self.stats['is_rtsp']:
                    # RTSP断线自动重连
                    print(f"[WARN] 摄像头 {self.camera_id} 读取失败，尝试重连...")
                    time.sleep(2)
                    self.cap.release()
                    self.cap = cv2.VideoCapture(self.source)
                    if not self.cap.isOpened():
                        self.running = False
                        break
                    continue
                else:
                    # 本地文件读完
                    print(f"[INFO] 本地视频 {self.camera_id} 播放结束")
                    self.running = False
                    break

            self.frame_count += 1
            self.stats['total_frames_read'] += 1
            fps_counter += 1

            # FPS 计算（每1秒更新）
            current_time = time.time()
            if current_time - last_frame_time >= 1.0:
                self.stats['fps'] = fps_counter
                fps_counter = 0
                last_frame_time = current_time

            # 跳帧策略：每 skip_frame 帧取1帧送入推理
            if self.frame_count % self.skip_frame == 0:
                # 尝试放入队列（非阻塞，队列满则丢弃）
                try:
                    self.frame_queue.put_nowait({
                        'camera_id': self.camera_id,
                        'frame': frame.copy(),
                        'frame_id': self.frame_count,
                        'timestamp': time.time()
                    })
                    self.stats['frames_sent_to_model'] += 1
                except queue.Full:
                    self.dropped_frames += 1
                    self.stats['dropped_frames'] += 1

        self.cap.release()

    def get_frame(self, timeout=1.0):
        """获取一帧（带超时）"""
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self):
        """停止读取"""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2)
        if self.cap and self.cap.isOpened():
            self.cap.release()

    def get_stats(self):
        """获取统计信息"""
        return self.stats.copy()


# ============================================================
# YOLO 行人检测器
# ============================================================

class YOLOPersonDetector:
    """
    改进 YOLOv8 行人检测器
    - 加载自定义 MixConv+BiFPN 模型
    - 支持批量推理
    - 输出检测框 + 置信度 + ROI裁剪
    """

    def __init__(self, model_path=None, conf_thres=0.25, iou_thres=0.45):
        """
        :param model_path: 模型权重路径（None则使用预训练）
        :param conf_thres: 置信度阈值
        :param iou_thres: NMS IoU阈值
        """
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 加载模型
        self.model = None
        self._load_model(model_path)

        # 输入尺寸
        self.imgsz = 640

        # 统计
        self.inference_times = []

        print(f"[INFO] YOLO 检测器初始化完成, 设备: {self.device}")

    def _load_model(self, model_path):
        """加载 YOLO 模型 (ultralytics 格式, 直接 YOLO() 加载)"""
        try:
            from ultralytics import YOLO
            if model_path and Path(model_path).exists():
                self.model = YOLO(str(model_path))
                print(f"[INFO] 加载 YOLO 检测模型: {model_path}")
            else:
                self.model = YOLO('yolov8n.pt')
                print("[INFO] 使用原生 YOLOv8n 预训练模型")
            self.model.to(self.device)
        except Exception as e:
            print(f"[WARN] YOLO 模型加载失败: {e}")
            self.model = None
            self._init_opencv_detector()

    def _init_opencv_detector(self):
        """初始化 OpenCV DNN 检测器（后备方案）"""
        self.net = None
        try:
            # 尝试加载 YOLO ONNX
            onnx_path = Path(__file__).parent.parent / 'weights' / 'yolov8n.onnx'
            if onnx_path.exists():
                self.net = cv2.dnn.readNetFromONNX(str(onnx_path))
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                print("[INFO] 使用 OpenCV DNN + ONNX 模型")
            else:
                print("[INFO] 使用 OpenCV HOG 行人检测器")
                self.hog = cv2.HOGDescriptor()
                self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
                self.net = None
        except Exception as e:
            print(f"[WARN] OpenCV 检测器初始化失败: {e}")
            self.net = None

    def detect(self, frame):
        """
        检测单帧中的行人 (ultralytics YOLO)
        :param frame: BGR 图像
        :return: 检测结果列表 [{bbox, confidence, class_id, roi}]
        """
        start_time = time.time()
        detections = []
        h, w = frame.shape[:2]

        # 方法1: ultralytics YOLO
        if self.model is not None:
            try:
                res = self.model.predict(
                    frame, imgsz=self.imgsz, conf=self.conf_thres,
                    iou=self.iou_thres, verbose=False)[0]
                boxes = res.boxes
                if boxes is not None:
                    for b in boxes:
                        if int(b.cls[0]) != 0:  # 仅保留 person 类
                            continue
                        x1, y1, x2, y2 = b.xyxy[0].tolist()
                        x1, y1 = max(0.0, x1), max(0.0, y1)
                        x2, y2 = min(float(w), x2), min(float(h), y2)
                        if x2 - x1 < 10 or y2 - y1 < 20:
                            continue
                        detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'confidence': float(b.conf[0]),
                            'class_id': 0,
                            'roi': frame[int(y1):int(y2), int(x1):int(x2)],
                        })
            except Exception as e:
                print(f"[WARN] YOLO 推理异常: {e}")
                detections = self._detect_fallback(frame)

        # 方法2: OpenCV DNN
        elif self.net is not None:
            try:
                detections = self._detect_opencv(frame)
            except Exception as e:
                detections = self._detect_fallback(frame)

        # 方法3: HOG 后备
        else:
            detections = self._detect_fallback(frame)

        # 计算耗时
        elapsed = (time.time() - start_time) * 1000
        self.inference_times.append(elapsed)
        if len(self.inference_times) > 100:
            self.inference_times = self.inference_times[-100:]

        return detections

    def _preprocess(self, frame):
        """图像预处理"""
        img = cv2.resize(frame, (self.imgsz, self.imgsz))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(self.device)
        return img

    def _postprocess(self, outputs, orig_size):
        """后处理：解码预测 + NMS"""
        h_orig, w_orig = orig_size
        detections = []

        # 处理分类和回归输出
        cls_scores, bbox_preds = outputs if isinstance(outputs, tuple) else (outputs, None)

        if isinstance(cls_scores, list):
            # 多尺度输出
            for cls_map, bbox_map in zip(cls_scores, bbox_preds if isinstance(bbox_preds, list) else [bbox_preds]):
                dets = self._decode_single_map(cls_map, bbox_map, (h_orig, w_orig))
                detections.extend(dets)
        elif isinstance(cls_scores, torch.Tensor):
            # 单尺度输出
            dets = self._decode_single_map(cls_scores, bbox_preds, (h_orig, w_orig))
            detections.extend(dets)

        # NMS
        detections = self._nms(detections)

        # 裁剪 ROI
        for det in detections:
            det['roi'] = self._crop_roi(frame, det['bbox'])

        return detections

    def _decode_single_map(self, cls_map, bbox_map, orig_size):
        """解码单尺度特征图"""
        h_orig, w_orig = orig_size
        detections = []

        if cls_map.dim() == 4:
            cls_map = cls_map.squeeze(0)
        if bbox_map.dim() == 4:
            bbox_map = bbox_map.squeeze(0)

        _, feat_h, feat_w = cls_map.shape
        stride_h = self.imgsz // feat_h
        stride_w = self.imgsz // feat_w

        with torch.no_grad():
            scores = cls_map.sigmoid()

            for y in range(feat_h):
                for x in range(feat_w):
                    score = scores[0, y, x].item()
                    if score < self.conf_thres:
                        continue

                    # 获取边界框预测
                    if bbox_map.shape[0] >= 4:
                        l, t, r, b = bbox_map[:4, y, x].tolist()
                    else:
                        l, t, r, b = 0, 0, 0, 0

                    # 转换到原图坐标
                    cx = (x + 0.5) * stride_w
                    cy = (y + 0.5) * stride_h

                    x1 = max(0, (cx - l * stride_w) * w_orig / self.imgsz)
                    y1 = max(0, (cy - t * stride_h) * h_orig / self.imgsz)
                    x2 = min(w_orig, (cx + r * stride_w) * w_orig / self.imgsz)
                    y2 = min(h_orig, (cy + b * stride_h) * h_orig / self.imgsz)

                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': score,
                        'class_id': 0
                    })

        return detections

    def _detect_fallback(self, frame):
        """后备行人检测（HOG）"""
        detections = []
        try:
            rects, weights = self.hog.detectMultiScale(
                frame, winStride=(4, 4), padding=(8, 8), scale=1.05
            )

            for (x, y, w, h), weight in zip(rects, weights):
                if weight > 0.5:  # 置信度过滤
                    detections.append({
                        'bbox': [float(x), float(y), float(x + w), float(y + h)],
                        'confidence': float(weight),
                        'class_id': 0,
                        'roi': cv2.resize(frame[y:y+h, x:x+w], (128, 256))
                    })
        except Exception:
            pass

        return detections

    def _detect_opencv(self, frame):
        """OpenCV DNN 检测"""
        detections = []
        h, w = frame.shape[:2]

        try:
            blob = cv2.dnn.blobFromImage(frame, 1/255.0, (640, 640), swapRB=True)
            self.net.setInput(blob)
            outputs = self.net.forward(self.net.getUnconnectedOutLayersNames())

            # 简化解析
            for output in outputs:
                if output is not None:
                    output = output.reshape(-1, output.shape[-1] if output.ndim == 3 else output.shape[-1])
                    # 仅保留person类(class 0)
                    for i in range(output.shape[0]):
                        row = output[i]
                        if len(row) >= 5:
                            cx, cy, bw, bh = row[:4]
                            score = row[4:].max() if len(row) > 4 else 0
                            class_id = row[4:].argmax() if len(row) > 4 else 0

                            if score > self.conf_thres and class_id == 0:
                                x1 = max(0, (cx - bw/2) * w)
                                y1 = max(0, (cy - bh/2) * h)
                                x2 = min(w, (cx + bw/2) * w)
                                y2 = min(h, (cy + bh/2) * h)

                                detections.append({
                                    'bbox': [x1, y1, x2, y2],
                                    'confidence': float(score),
                                    'class_id': int(class_id),
                                    'roi': self._crop_roi(frame, [x1, y1, x2, y2])
                                })
        except Exception as e:
            print(f"[WARN] OpenCV 检测异常: {e}")

        return detections

    def _crop_roi(self, frame, bbox):
        """裁剪行人 ROI 并缩放至 256×128"""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(max(0, min(v, max(w, h)))) for v in bbox]

        if x2 > x1 and y2 > y1:
            roi = frame[y1:y2, x1:x2]
            if roi.size > 0:
                # 统一缩放至 ReID 输入尺寸（256高 x 128宽）
                roi_resized = cv2.resize(roi, (128, 256))
                return roi_resized

        # 返回空白 ROI
        return np.zeros((256, 128, 3), dtype=np.uint8)

    def _nms(self, detections):
        """非极大值抑制"""
        if not detections:
            return []

        boxes = np.array([d['bbox'] for d in detections])
        scores = np.array([d['confidence'] for d in detections])

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            if order.size == 1:
                break

            remaining = order[1:]

            # 计算 IoU
            xx1 = np.maximum(x1[i], x1[remaining])
            yy1 = np.maximum(y1[i], y1[remaining])
            xx2 = np.minimum(x2[i], x2[remaining])
            yy2 = np.minimum(y2[i], y2[remaining])

            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            iou = inter / (areas[i] + areas[remaining] - inter + 1e-6)

            order = remaining[iou <= self.iou_thres]

        return [detections[i] for i in keep]

    def get_avg_inference_time(self):
        """获取平均推理耗时"""
        if self.inference_times:
            return np.mean(self.inference_times)
        return 0.0


# ============================================================
# 多路视频流管理器
# ============================================================

class MultiCameraManager:
    """
    多路摄像头管理器
    - 管理多路视频流
    - 轮询检测 + 结果分发
    """

    def __init__(self, detector, skip_frame=2):
        """
        :param detector: YOLOPersonDetector 实例
        :param skip_frame: 跳帧间隔
        """
        self.detector = detector
        self.skip_frame = skip_frame
        self.cameras = {}  # camera_id -> CameraStreamReader
        self.results_queue = queue.Queue(maxsize=100)
        self.running = False

        # 全局统计
        self.global_stats = {
            'total_detections': 0,
            'processed_frames': 0,
            'avg_fps': 0.0,
            'start_time': None
        }

    def add_camera(self, source, camera_id):
        """添加摄像头"""
        reader = CameraStreamReader(source, camera_id, skip_frame=self.skip_frame)
        if reader.start():
            self.cameras[camera_id] = reader
            return True
        return False

    def remove_camera(self, camera_id):
        """移除摄像头"""
        if camera_id in self.cameras:
            self.cameras[camera_id].stop()
            del self.cameras[camera_id]
            return True
        return False

    def start_detection_loop(self, callback=None):
        """
        启动检测循环
        :param callback: 检测结果回调函数
        """
        self.running = True
        self.global_stats['start_time'] = time.time()

        print(f"\n[INFO] 启动多路检测循环，共 {len(self.cameras)} 路摄像头")

        try:
            while self.running:
                all_frames = []

                for cam_id, reader in self.cameras.items():
                    if reader.running:
                        frame_data = reader.get_frame(timeout=0.5)
                        if frame_data is not None:
                            all_frames.append(frame_data)

                if not all_frames:
                    time.sleep(0.01)
                    continue

                # 对每帧进行检测
                for frame_data in all_frames:
                    camera_id = frame_data['camera_id']
                    frame = frame_data['frame']

                    detections = self.detector.detect(frame)

                    self.global_stats['processed_frames'] += 1
                    self.global_stats['total_detections'] += len(detections)

                    result = {
                        'camera_id': camera_id,
                        'frame_id': frame_data['frame_id'],
                        'timestamp': frame_data['timestamp'],
                        'detections': detections,
                        'frame': frame  # 保留用于后续处理
                    }

                    # 放入结果队列
                    try:
                        self.results_queue.put_nowait(result)
                    except queue.Full:
                        # 队列满则丢弃旧结果
                        try:
                            self.results_queue.get_nowait()
                            self.results_queue.put_nowait(result)
                        except queue.Empty:
                            pass

                    # 回调
                    if callback:
                        try:
                            callback(result)
                        except Exception as e:
                            print(f"[WARN] 回调执行异常: {e}")

                # 更新全局 FPS
                elapsed = time.time() - self.global_stats['start_time']
                if elapsed > 0:
                    self.global_stats['avg_fps'] = self.global_stats['processed_frames'] / elapsed

        except KeyboardInterrupt:
            self.stop()

    def get_result(self, timeout=0.1):
        """获取检测结果"""
        try:
            return self.results_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self):
        """停止所有摄像头"""
        self.running = False
        for cam_id, reader in self.cameras.items():
            reader.stop()
        print(f"[INFO] 已停止 {len(self.cameras)} 路摄像头")

    def get_all_stats(self):
        """获取所有统计信息"""
        stats = {
            'global': self.global_stats,
            'cameras': {}
        }
        for cam_id, reader in self.cameras.items():
            stats['cameras'][cam_id] = reader.get_stats()
        return stats


# ============================================================
# 独立调试入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="多路视频流检测调试工具")
    parser.add_argument("--sources", nargs='+',
                        default=["test_video/hotel_raw/camera_01.mp4"],
                        help="视频源路径列表")
    parser.add_argument("--camera_ids", nargs='+',
                        default=["CAM_001"],
                        help="摄像头编号列表")
    parser.add_argument("--model", type=str, default=None,
                        help="YOLO模型权重路径")
    parser.add_argument("--skip_frame", type=int, default=2,
                        help="跳帧间隔")
    parser.add_argument("--display", action="store_true",
                        help="显示检测窗口")
    parser.add_argument("--duration", type=int, default=30,
                        help="运行时长（秒）")

    args = parser.parse_args()

    print("=" * 60)
    print("多路视频流检测调试工具")
    print("=" * 60)

    # 初始化检测器
    detector = YOLOPersonDetector(model_path=args.model)

    # 初始化管理器
    manager = MultiCameraManager(detector, skip_frame=args.skip_frame)

    # 添加摄像头
    for i, source in enumerate(args.sources):
        cam_id = args.camera_ids[i] if i < len(args.camera_ids) else f"CAM_{i+1:03d}"
        success = manager.add_camera(source, cam_id)
        if success:
            print(f"  ✓ {cam_id}: {source}")
        else:
            print(f"  ✗ {cam_id}: {source} (启动失败)")

    if not manager.cameras:
        print("\n[ERROR] 无可用摄像头，退出")
        return

    # 调试模式：显示检测窗口
    if args.display:
        print(f"\n[INFO] 启动交互式调试模式，按 'q' 退出...")

        # 设置回调显示
        detection_log = []

        def display_callback(result):
            frame = result['frame']
            detections = result['detections']
            cam_id = result['camera_id']

            # 绘制检测框
            for det in detections:
                x1, y1, x2, y2 = [int(v) for v in det['bbox']]
                color = (0, 255, 0)  # 绿色
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # 标签
                label = f"Person: {det['confidence']:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # 显示信息
            info = f"{cam_id} | Det: {len(detections)} | " \
                   f"Frames: {detector.stats.get('frames_sent_to_model', 0)}"
            cv2.putText(frame, info, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # 显示
            cv2.imshow(f"Camera: {cam_id}", frame)

            # 记录
            detection_log.append({
                'camera_id': cam_id,
                'num_detections': len(detections),
                'timestamp': time.time()
            })

        # 启动检测循环（带回调）
        import threading

        def run_detection():
            manager.start_detection_loop(callback=display_callback)

        det_thread = threading.Thread(target=run_detection, daemon=True)
        det_thread.start()

        # 等待退出或超时
        start_time = time.time()
        while True:
            if time.time() - start_time > args.duration:
                print(f"\n[INFO] 运行时长 {args.duration}s 已到，退出")
                break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n[INFO] 用户按 'q' 退出")
                break
            time.sleep(0.01)

        cv2.destroyAllWindows()
        manager.stop()

        # 输出统计
        print(f"\n[统计] 检测日志记录: {len(detection_log)}")
        if detection_log:
            avg_detections = np.mean([d['num_detections'] for d in detection_log])
            print(f"[统计] 平均每帧检测人数: {avg_detections:.2f}")

    else:
        # 无界面模式
        print(f"\n[INFO] 启动静默调试模式，运行 {args.duration}s...")

        # 统计变量
        detection_counts = defaultdict(list)
        frame_count = 0

        def stats_callback(result):
            nonlocal frame_count
            frame_count += 1
            cam_id = result['camera_id']
            detection_counts[cam_id].append(len(result['detections']))

        # 启动
        import threading
        det_thread = threading.Thread(
            target=manager.start_detection_loop,
            kwargs={'callback': stats_callback},
            daemon=True
        )
        det_thread.start()

        # 等待
        time.sleep(args.duration)
        manager.stop()

        # 输出统计
        print("\n" + "=" * 60)
        print("多路检测调试统计结果")
        print("=" * 60)

        stats = manager.get_all_stats()
        for cam_id, cam_stats in stats['cameras'].items():
            print(f"\n  摄像头: {cam_id}")
            print(f"    总读取帧数: {cam_stats['total_frames_read']}")
            print(f"    送入模型帧数: {cam_stats['frames_sent_to_model']}")
            print(f"    丢弃帧数: {cam_stats['dropped_frames']}")
            print(f"    实时 FPS: {cam_stats['fps']}")
            print(f"    分辨率: {cam_stats['resolution']}")

            if cam_id in detection_counts:
                counts = detection_counts[cam_id]
                if counts:
                    print(f"    平均检测人数/帧: {np.mean(counts):.2f}")
                    print(f"    最大检测人数/帧: {np.max(counts)}")

        print(f"\n  全局统计:")
        print(f"    处理帧总数: {stats['global']['processed_frames']}")
        print(f"    检测目标总数: {stats['global']['total_detections']}")
        print(f"    平均处理FPS: {stats['global']['avg_fps']:.2f}")
        print(f"    平均推理耗时: {detector.get_avg_inference_time():.2f}ms")


if __name__ == '__main__':
    main()