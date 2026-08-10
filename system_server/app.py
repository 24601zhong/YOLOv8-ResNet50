# -*- coding: utf-8 -*-
"""
酒店异常人员实时监控识别系统 - Flask 主应用（完整版）
功能模块：
  1. 入住登记与 MySQL 数据交互
  2. 多路视频流 + YOLO 检测 + ReID 匹配
  3. MJPEG 视频推流 + 绿/红框叠加
  4. 异常预警触发（弹窗 + 蜂鸣 + 截图 + 日志）
  5. 可视化仪表板与历史记录

启动：python app.py --host 0.0.0.0 --port 5000
"""

import os
import sys
import json
import time
import threading
import argparse
import base64
import queue
import numpy as np
from datetime import datetime
from pathlib import Path

import cv2
import torch
from flask import Flask, render_template, request, jsonify, Response
from flask_cors import CORS

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from system_server.db_mysql import HotelDatabase
from system_server.video_stream import YOLOPersonDetector, MultiCameraManager
from system_server.reid_match import ReidMatchingPipeline
from system_server.alert_manager import AlertManager


# ============================================================
# Flask 应用初始化
# ============================================================

app = Flask(__name__)
CORS(app)

app.config['SECRET_KEY'] = 'hotel_security_system_2026'
app.config['UPLOAD_FOLDER'] = str(PROJECT_ROOT / 'output' / 'alert_screenshots')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

Path(app.config['UPLOAD_FOLDER']).mkdir(parents=True, exist_ok=True)

# 全局实例
db = None
yolo_detector = None
reid_pipeline = None
alert_mgr = None
camera_mgr = None
system_running = False

# MJPEG 帧缓冲区
frame_buffers = {}  # camera_id -> {'frame', 'timestamp', 'detections'}
frame_lock = threading.Lock()

# 系统统计
system_stats = {
    'fps': 0.0, 'online_cameras': 0, 'total_detections': 0,
    'matched_persons': 0, 'alerts_today': 0,
    'start_time': time.time(), 'is_running': False
}


# ============================================================
# 初始化系统组件
# ============================================================

def init_system(db_config, yolo_model_path, reid_model_path):
    """初始化全部系统组件"""
    global db, yolo_detector, reid_pipeline, alert_mgr, camera_mgr, system_running

    print("\n" + "=" * 60)
    print("酒店异常人员实时监控识别系统 - 初始化")
    print("=" * 60)

    # 1. 数据库
    db = HotelDatabase(**db_config)
    if db.test_connection():
        print("[1/4] ✓ MySQL 数据库连接成功")
    else:
        print("[1/4] ⚠ MySQL 连接失败，使用演示模式")

    # 2. YOLO 检测器
    yolo_detector = YOLOPersonDetector(model_path=yolo_model_path)
    print(f"[2/4] ✓ YOLO 检测器就绪 (设备: {yolo_detector.device})")

    # 3. ReID 流水线
    reid_pipeline = ReidMatchingPipeline(model_path=reid_model_path, db_config=db_config)
    print(f"[3/4] ✓ ReID 流水线就绪 (特征库: {reid_pipeline.feature_db.size()} 人)")

    # 4. 预警管理器
    alert_mgr = AlertManager(output_dir=app.config['UPLOAD_FOLDER'], db_config=db_config)
    print("[4/4] ✓ 预警管理器就绪")

    system_running = True
    system_stats['is_running'] = True

    print("\n" + "=" * 60)
    print("系统初始化完成！")
    print("=" * 60 + "\n")


def add_camera(source, camera_id):
    """添加摄像头"""
    global camera_mgr
    if camera_mgr is None:
        camera_mgr = MultiCameraManager(yolo_detector, skip_frame=2)
    return camera_mgr.add_camera(source, camera_id)


def detection_loop():
    """
    主检测循环
    - 从 camera_mgr 获取检测结果
    - 对每个行人进行 ReID 匹配
    - 触发预警
    - 更新 MJPEG 帧缓冲区
    """
    global system_stats

    while system_running:
        if camera_mgr is None:
            time.sleep(0.1)
            continue

        result = camera_mgr.get_result(timeout=0.1)
        if result is None:
            continue

        camera_id = result['camera_id']
        frame = result['frame']
        raw_detections = result['detections']

        processed_detections = []

        for det in raw_detections:
            roi = det.get('roi')
            if roi is not None and roi.size > 0:
                match_result = reid_pipeline.process(roi)
                det['is_matched'] = match_result['is_matched']
                det['is_anomaly'] = match_result['is_anomaly']
                det['similarity'] = match_result['similarity']
                det['person_name'] = match_result['person_info'].get('name', 'Guest')
                det['person_room'] = match_result['person_info'].get('room_num', '')

                if match_result['is_anomaly']:
                    alert_mgr.trigger_alert(
                        camera_id=camera_id, frame=frame,
                        similarity=match_result['similarity'],
                        person_id=match_result.get('person_id', -1),
                        person_info=match_result['person_info']
                    )
                    system_stats['alerts_today'] += 1
                else:
                    system_stats['matched_persons'] += 1

            processed_detections.append(det)

        # 更新 MJPEG 帧
        update_mjpeg_frame(camera_id, frame, processed_detections)
        system_stats['total_detections'] += len(processed_detections)
        system_stats['online_cameras'] = len(frame_buffers)


def update_mjpeg_frame(camera_id, frame, detections):
    """更新 MJPEG 帧缓冲区"""
    annotated = frame.copy()

    for det in detections:
        bbox = det.get('bbox', [0, 0, 0, 0])
        x1, y1, x2, y2 = [int(v) for v in bbox]
        conf = det.get('confidence', 0)
        is_anomaly = det.get('is_anomaly', False)

        color = (0, 0, 255) if is_anomaly else (0, 255, 0)
        label = f"ALERT:{conf:.2f}" if is_anomaly else f"{det.get('person_name','Guest')} R:{det.get('person_room','--')}"

        # 绘制检测框
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)

        # 标签背景
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 5, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # 底部信息栏
    h = annotated.shape[0]
    info = f"{camera_id} | {datetime.now().strftime('%H:%M:%S')} | FPS: {system_stats['fps']:.1f}"
    cv2.putText(annotated, info, (10, h - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    with frame_lock:
        frame_buffers[camera_id] = {
            'frame': annotated, 'timestamp': time.time(), 'detections': detections
        }


# ============================================================
# MJPEG 视频流端点
# ============================================================

@app.route('/video_feed/<camera_id>')
def video_feed(camera_id):
    """MJPEG 视频流"""
    def generate():
        boundary = b'--frame\r\n'
        content = b'Content-Type: image/jpeg\r\n\r\n'
        while True:
            with frame_lock:
                buf = frame_buffers.get(camera_id)
            if buf and buf.get('frame') is not None:
                _, jpeg = cv2.imencode('.jpg', buf['frame'], [cv2.IMWRITE_JPEG_QUALITY, 85])
                yield boundary + content + jpeg.tobytes() + b'\r\n'
            time.sleep(0.01)

    return Response(generate(),
                    content_type='multipart/x-mixed-replace; boundary=frame')


# ============================================================
# 页面路由
# ============================================================

@app.route('/')
def dashboard():
    """仪表板"""
    stats = db.get_statistics() if db else {
        'total_persons': 0, 'unhandled_alerts': 0,
        'total_alerts': 0, 'handled_alerts': 0, 'today_alerts': 0
    }
    recent = []
    if db:
        recent = db.get_all_alerts(limit=5)
    return render_template('dashboard.html', stats=stats, recent_alerts=recent)


@app.route('/register')
def register_page():
    return render_template('register.html')


@app.route('/monitor')
def monitor_page():
    cameras = [
        {'id': 'CAM_001', 'name': '大堂入口', 'stream': '/video_feed/CAM_001'},
        {'id': 'CAM_002', 'name': '电梯厅', 'stream': '/video_feed/CAM_002'},
        {'id': 'CAM_003', 'name': '走廊A区', 'stream': '/video_feed/CAM_003'},
        {'id': 'CAM_004', 'name': '后门出口', 'stream': '/video_feed/CAM_004'}
    ]
    return render_template('monitor_v2.html', cameras=cameras)


@app.route('/alerts')
def alerts_page():
    status_filter = request.args.get('status')
    alerts = []
    if db:
        alerts = db.get_all_alerts(
            status=int(status_filter) if status_filter else None
        )
    return render_template('alerts.html', alerts=alerts)


@app.route('/persons')
def persons_page():
    search = request.args.get('search', '')
    persons = []
    if db:
        persons = db.search_persons(search) if search else db.get_all_persons()
    return render_template('persons.html', persons=persons, search=search)


# ============================================================
# API 路由
# ============================================================

@app.route('/api/stats')
def api_stats():
    return jsonify({
        'system': system_stats,
        'db': db.get_statistics() if db else {},
        'cameras': [{'id': k, 'active': True} for k in frame_buffers.keys()],
        'alert_stats': alert_mgr.get_stats() if alert_mgr else {}
    })


@app.route('/api/cameras')
def api_cameras():
    return jsonify([
        {'id': 'CAM_001', 'name': '大堂入口', 'location': '一楼大堂'},
        {'id': 'CAM_002', 'name': '电梯厅', 'location': '一楼电梯厅'},
        {'id': 'CAM_003', 'name': '走廊A区', 'location': '二楼走廊'},
        {'id': 'CAM_004', 'name': '后门出口', 'location': '后门'}
    ])


@app.route('/api/detect', methods=['POST'])
def api_detect():
    """单帧检测 API"""
    data = request.get_json()
    camera_id = data.get('camera_id', 'CAM_001')
    frame_b64 = data.get('frame', '')

    if not frame_b64:
        return jsonify({'error': '缺少图像帧'}), 400

    try:
        img_data = base64.b64decode(
            frame_b64.split(',')[1] if ',' in frame_b64 else frame_b64
        )
        np_arr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({'error': '图像解码失败'}), 400

        # YOLO 检测
        detections = yolo_detector.detect(frame)

        # ReID 匹配（取第一个行人）
        result = {'camera_id': camera_id, 'detections': []}
        for det in detections:
            roi = det.get('roi')
            if roi is not None and roi.size > 0:
                match = reid_pipeline.process(roi)
                det['is_matched'] = match['is_matched']
                det['is_anomaly'] = match['is_anomaly']
                det['similarity'] = match['similarity']
                det['person_name'] = match['person_info'].get('name', '')

                if match['is_anomaly']:
                    alert_mgr.trigger_alert(camera_id, frame, match['similarity'])

            result['detections'].append(det)

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/register', methods=['POST'])
def api_register():
    """入住登记 API"""
    data = request.get_json() or request.form.to_dict()
    name = data.get('name', '').strip()
    id_card = data.get('id_card', '').strip()
    room_num = data.get('room_num', '').strip()
    face_b64 = data.get('face_image', '')

    if not all([name, id_card, room_num]):
        return jsonify({'success': False, 'message': '请填写完整信息'}), 400

    if len(id_card) != 18:
        return jsonify({'success': False, 'message': '身份证号格式错误'}), 400

    if db:
        existing = db.get_person_by_id_card(id_card)
        if existing:
            return jsonify({'success': False, 'message': '身份证已登记'}), 409

    feature_vec = None
    face_path = None

    if face_b64:
        try:
            img_data = base64.b64decode(
                face_b64.split(',')[1] if ',' in face_b64 else face_b64
            )
            np_arr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if img is not None:
                ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                face_path = os.path.join(app.config['UPLOAD_FOLDER'], f"reg_{id_card}_{ts}.jpg")
                cv2.imwrite(face_path, img)

                feature = reid_pipeline.extractor.extract(img)
                feature_vec = json.dumps(feature.tolist())

                person_id = db.insert_person(name, id_card, room_num,
                                            face_img_path=face_path)
                # 刷新特征库
                reid_pipeline.feature_db.load_from_mysql()

                return jsonify({
                    'success': True, 'message': '登记成功',
                    'person_id': person_id, 'feature_extracted': True
                })
        except Exception as e:
            return jsonify({'success': False, 'message': str(e)}), 500

    person_id = db.insert_person(name, id_card, room_num)
    return jsonify({'success': True, 'person_id': person_id})


@app.route('/api/alerts')
def api_alerts():
    """预警列表 API"""
    limit = request.args.get('limit', 50, type=int)
    alerts = []
    if db:
        alerts = db.get_all_alerts(limit=limit)
    return jsonify(alerts)


@app.route('/api/alerts/<int:log_id>/handle', methods=['POST'])
def api_handle_alert(log_id):
    data = request.get_json()
    status = data.get('status', 1)
    ok = db.update_alert_status(log_id, status) if db else False
    return jsonify({'success': ok})


@app.route('/api/alerts/<int:log_id>', methods=['DELETE'])
def api_delete_alert(log_id):
    ok = db.delete_alert(log_id) if db else False
    return jsonify({'success': ok})


@app.route('/api/persons/<int:person_id>/checkout', methods=['POST'])
def api_checkout(person_id):
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    ok = db.update_person(person_id, check_out_time=now) if db else False
    return jsonify({'success': ok, 'check_out_time': now})


@app.route('/api/persons/<int:person_id>', methods=['DELETE'])
def api_delete_person(person_id):
    ok = db.delete_person(person_id) if db else False
    if ok:
        reid_pipeline.feature_db.load_from_mysql()
    return jsonify({'success': ok})


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="酒店异常人员实时监控识别系统")
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--model', default='Hotel_Exp/output/reid_train_log/best_reid.pt')
    parser.add_argument('--yolo_model', default=None)
    parser.add_argument('--video', default=None, help='测试视频路径')
    parser.add_argument('--rtsp', action='store_true', help='使用 RTSP 摄像头')
    parser.add_argument('--db_host', default='localhost')
    parser.add_argument('--db_port', type=int, default=3306)
    parser.add_argument('--db_user', default='root')
    parser.add_argument('--db_password', default='root')
    parser.add_argument('--db_name', default='hotel_security')

    args = parser.parse_args()

    db_config = {
        'host': args.db_host, 'port': args.db_port,
        'user': args.db_user, 'password': args.db_password,
        'database': args.db_name
    }

    # 初始化系统
    init_system(db_config, args.yolo_model, args.model)

    # 添加摄像头
    if args.video:
        add_camera(args.video, 'CAM_001')
    elif args.rtsp:
        add_camera('rtsp://localhost:554/stream', 'CAM_001')
    else:
        # 默认使用本地摄像头或测试视频
        test_video = PROJECT_ROOT / 'test_video' / 'hotel_raw' / 'demo.mp4'
        if test_video.exists():
            add_camera(str(test_video), 'CAM_001')
        else:
            add_camera(0, 'CAM_001')  # 本地摄像头

    # 启动检测线程
    det_thread = threading.Thread(target=detection_loop, daemon=True)
    det_thread.start()

    # 启动摄像头读取
    if camera_mgr:
        read_thread = threading.Thread(
            target=camera_mgr.start_detection_loop, daemon=True
        )
        read_thread.start()

    # 启动 Flask
    print(f"\n{'='*60}")
    print(f"  系统启动! 浏览器访问: http://localhost:{args.port}")
    print(f"  MJPEG 流: http://localhost:{args.port}/video_feed/CAM_001")
    print(f"{'='*60}\n")

    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == '__main__':
    main()