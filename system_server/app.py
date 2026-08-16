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
from flask import Flask, render_template, request, jsonify, Response, send_from_directory
from flask_cors import CORS

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from system_server.db_mysql import HotelDatabase
from system_server.video_stream import YOLOPersonDetector, MultiCameraManager, build_rtsp_url
from system_server.reid_match import ReidMatchingPipeline
from system_server.alert_manager import AlertManager
from system_server.face_match import (
    FaceFeatureDatabase, FaceMatchingPipeline, FusedMatchingPipeline, align_face,
)


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
fused_pipeline = None
alert_mgr = None
camera_mgr = None
system_running = False

# MJPEG 帧缓冲区
frame_buffers = {}  # camera_id -> {'frame', 'timestamp', 'detections'}
frame_lock = threading.Lock()

# 摄像头注册表 (运行时管理 + JSON 持久化)
camera_registry = {}  # camera_id -> {camera_id, name, location, source_type, source}
camera_registry_lock = threading.Lock()

# 最新检测结果缓存 (camera_id -> [detections])，供预览线程异步叠加
latest_detections = {}
detection_lock = threading.Lock()

CAMERAS_CONFIG = PROJECT_ROOT / 'config' / 'cameras.json'

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
    global db, yolo_detector, reid_pipeline, fused_pipeline, alert_mgr, camera_mgr, system_running

    print("\n" + "=" * 60)
    print("酒店异常人员实时监控识别系统 - 初始化")
    print("=" * 60)

    # 1. 数据库
    db = HotelDatabase(**db_config)
    if db.test_connection():
        print("[1/5] [OK] MySQL 数据库连接成功")
    else:
        print("[1/5] [WARN] MySQL 连接失败，使用演示模式")

    # 2. YOLO 检测器
    yolo_detector = YOLOPersonDetector(model_path=yolo_model_path)
    print(f"[2/5] [OK] YOLO 检测器就绪 (设备: {yolo_detector.device})")

    # 3. ReID 流水线
    reid_pipeline = ReidMatchingPipeline(model_path=reid_model_path, db_config=db_config)
    print(f"[3/5] [OK] ReID 流水线就绪 (特征库: {reid_pipeline.feature_db.size()} 人)")

    # 4. 人脸流水线 + 融合 (人脸优先 + ReID 兜底)
    face_db = FaceFeatureDatabase(db_config=db_config)
    face_pipeline = FaceMatchingPipeline(face_db=face_db, threshold=0.4)
    fused_pipeline = FusedMatchingPipeline(face_pipeline=face_pipeline, reid_pipeline=reid_pipeline)
    print(f"[4/5] [OK] 融合流水线就绪 (人脸优先 + ReID 兜底, 人脸库: {face_db.size()} 人)")

    # 5. 预警管理器
    alert_mgr = AlertManager(output_dir=app.config['UPLOAD_FOLDER'], db_config=db_config)
    print("[5/5] [OK] 预警管理器就绪")

    system_running = True
    system_stats['is_running'] = True

    print("\n" + "=" * 60)
    print("系统初始化完成！")
    print("=" * 60 + "\n")


def load_cameras():
    """从 JSON 读取摄像头配置列表 (不直接注册, 由 main 逐个 add_camera)"""
    if not CAMERAS_CONFIG.exists():
        return []
    try:
        with open(CAMERAS_CONFIG, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get('cameras', []) if isinstance(data, dict) else data
    except Exception as e:
        print(f"[WARN] 摄像头配置加载失败: {e}")
        return []


def save_cameras():
    """持久化摄像头配置到 JSON"""
    try:
        CAMERAS_CONFIG.parent.mkdir(parents=True, exist_ok=True)
        with camera_registry_lock:
            cams = list(camera_registry.values())
        with open(CAMERAS_CONFIG, 'w', encoding='utf-8') as f:
            json.dump({'cameras': cams}, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[WARN] 摄像头配置保存失败: {e}")


def add_camera(source, camera_id, name=None, location='', source_type='local'):
    """添加摄像头并注册元数据"""
    global camera_mgr
    if camera_mgr is None:
        camera_mgr = MultiCameraManager(yolo_detector, skip_frame=2)
    ok = camera_mgr.add_camera(source, camera_id)
    if ok:
        with camera_registry_lock:
            camera_registry[camera_id] = {
                'camera_id': camera_id,
                'name': name or camera_id,
                'location': location,
                'source_type': source_type,
                'source': str(source),
            }
        save_cameras()
    return ok


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
            bbox = det.get('bbox')
            if bbox is not None:
                # 人脸优先 + ReID 兜底 融合匹配 (原分辨率帧 + 行人框)
                match_result = fused_pipeline.process(frame, bbox)
                det['is_matched'] = match_result['is_matched']
                det['is_anomaly'] = match_result['is_anomaly']
                det['similarity'] = match_result['similarity']
                det['person_name'] = match_result['person_info'].get('name', 'Guest')
                det['person_room'] = match_result['person_info'].get('room_num', '')
                det['matched_by'] = match_result.get('matched_by', 'reid')

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

        # 用检测帧直接标注写入 MJPEG 缓冲 (帧与框同步, 避免"最新画面+旧框"错位)
        update_mjpeg_frame(camera_id, frame, processed_detections)
        # 同时缓存检测结果 (供 snapshot 等复用)
        with detection_lock:
            latest_detections[camera_id] = processed_detections
        system_stats['total_detections'] += len(processed_detections)
        system_stats['online_cameras'] = len(frame_buffers)


def annotate_frame(frame, camera_id, detections):
    """在帧上叠加检测框/标签，返回标注后的帧（纯函数）"""
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

    return annotated


def update_mjpeg_frame(camera_id, frame, detections):
    """标注帧并写入 MJPEG 缓冲区"""
    annotated = annotate_frame(frame, camera_id, detections)
    with frame_lock:
        frame_buffers[camera_id] = {
            'frame': annotated, 'timestamp': time.time(), 'detections': detections
        }


def preview_loop():
    """兜底预览循环: 标注帧由 detection_loop 在检测完成后写入 (帧与框同步)。
    仅当某摄像头尚无标注帧、或标注帧超过 2 秒未更新时, 推一张最新裸帧兜底,
    避免黑屏/卡死, 同时不再把"最新画面"和"旧检测框"错位拼接。"""
    while system_running:
        if camera_mgr is None:
            time.sleep(0.1)
            continue

        try:
            for cam_id, reader in list(camera_mgr.cameras.items()):
                with frame_lock:
                    buf = frame_buffers.get(cam_id)
                stale = (buf is None or buf.get('frame') is None
                         or (time.time() - buf.get('timestamp', 0)) > 2.0)
                if not stale:
                    continue
                frame, _fid = reader.get_latest_frame()
                if frame is not None:
                    update_mjpeg_frame(cam_id, frame, [])
        except Exception as e:
            print(f"[WARN] 预览循环异常: {e}")

        time.sleep(0.03)


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


@app.route('/cameras')
def cameras_page():
    return render_template('cameras.html')


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
        for a in alerts:
            if a.get('screenshot_path'):
                a['screenshot_url'] = '/output/alert_screenshots/' + os.path.basename(a['screenshot_path'])
    return render_template('alerts.html', alerts=alerts)


@app.route('/persons')
def persons_page():
    search = request.args.get('search', '')
    persons = []
    if db:
        persons = db.search_persons(search) if search else db.get_all_persons()
        for p in persons:
            if p.get('face_img_path'):
                p['face_img_url'] = '/output/alert_screenshots/' + os.path.basename(p['face_img_path'])
    return render_template('persons.html', persons=persons, search=search)


@app.route('/output/<path:filename>')
def serve_output(filename):
    """服务 output/ 下的静态文件 (人员头像 / 预警截图)"""
    return send_from_directory(str(PROJECT_ROOT / 'output'), filename)


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
    """摄像头列表（真实连接 + 元数据）"""
    cameras = []
    with camera_registry_lock:
        reg = dict(camera_registry)
    readers = camera_mgr.cameras if camera_mgr else {}
    for cam_id, meta in reg.items():
        reader = readers.get(cam_id)
        cam = dict(meta)
        cam['id'] = cam_id  # 兼容前端旧字段
        cam['active'] = reader is not None and reader.running
        if reader is not None:
            stats = reader.get_stats()
            cam['resolution'] = stats.get('resolution')
            cam['fps'] = stats.get('fps')
            cam['is_rtsp'] = stats.get('is_rtsp')
            cam['is_camera'] = stats.get('is_camera')
        cameras.append(cam)
    return jsonify(cameras)


@app.route('/api/cameras', methods=['POST'])
def api_add_camera():
    """添加摄像头（本地 index 或 RTSP ip/port/user/pass）"""
    data = request.get_json() or {}
    name = (data.get('name') or '').strip()
    location = (data.get('location') or '').strip()
    source_type = data.get('source_type', 'local')

    if source_type == 'rtsp':
        ip = (data.get('ip') or '').strip()
        if not ip:
            return jsonify({'success': False, 'message': '缺少 IP 地址'}), 400
        try:
            port = int(data.get('port', 554))
        except (TypeError, ValueError):
            port = 554
        username = (data.get('username') or '').strip() or None
        password = data.get('password') or None
        path = (data.get('path') or '').strip()
        source = build_rtsp_url(ip, port, username, password, path)
    else:
        try:
            source = int(data.get('index', 0))
        except (TypeError, ValueError):
            source = 0

    base_id = (data.get('camera_id') or '').strip() or f"CAM_{int(time.time() * 1000) % 100000:05d}"
    camera_id = base_id
    i = 1
    while camera_id in camera_registry:
        camera_id = f"{base_id}_{i}"
        i += 1

    ok = add_camera(source, camera_id, name=name, location=location, source_type=source_type)
    if ok:
        return jsonify({'success': True, 'camera_id': camera_id, 'message': '摄像头已添加'})
    return jsonify({'success': False, 'message': f'无法打开摄像头: {source}'}), 500


@app.route('/api/cameras/<camera_id>', methods=['DELETE'])
def api_remove_camera(camera_id):
    """移除摄像头"""
    global camera_mgr
    if camera_mgr is not None:
        camera_mgr.remove_camera(camera_id)
    with camera_registry_lock:
        removed = camera_registry.pop(camera_id, None) is not None
    with frame_lock:
        frame_buffers.pop(camera_id, None)
    with detection_lock:
        latest_detections.pop(camera_id, None)
    if removed:
        save_cameras()
    return jsonify({'success': True})


@app.route('/api/cameras/<camera_id>/snapshot')
def api_camera_snapshot(camera_id):
    """抓取摄像头最新帧（JPEG，供登记页抓拍/预览）"""
    reader = camera_mgr.cameras.get(camera_id) if camera_mgr else None
    if reader is None:
        return jsonify({'error': '摄像头不存在'}), 404
    frame, _ = reader.get_latest_frame()
    if frame is None:
        return jsonify({'error': '暂无画面'}), 503
    with detection_lock:
        dets = latest_detections.get(camera_id, [])
    annotated = annotate_frame(frame, camera_id, dets) if dets else frame
    _, jpeg = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return Response(jpeg.tobytes(), mimetype='image/jpeg')


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

        # 融合匹配 (人脸优先 + ReID 兜底)
        result = {'camera_id': camera_id, 'detections': []}
        for det in detections:
            bbox = det.get('bbox')
            if bbox is not None:
                match = fused_pipeline.process(frame, bbox)
                det['is_matched'] = match['is_matched']
                det['is_anomaly'] = match['is_anomaly']
                det['similarity'] = match['similarity']
                det['person_name'] = match['person_info'].get('name', '')
                det['matched_by'] = match.get('matched_by', 'reid')

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
    face_vec = None
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

                # 先裁出登记主体 (行人框), 与检测链路对齐
                person_crop = img
                if yolo_detector is not None:
                    try:
                        dets = yolo_detector.detect(img)
                        if dets:
                            bbox = max(dets, key=lambda d: (d['bbox'][2]-d['bbox'][0])*(d['bbox'][3]-d['bbox'][1]))['bbox']
                            x1, y1, x2, y2 = [int(v) for v in bbox]
                            if x2 > x1 and y2 > y1:
                                person_crop = img[y1:y2, x1:x2]
                    except Exception as e:
                        print(f"[WARN] 行人检测失败(回退整图): {e}")

                # ReID 全身特征 (兜底): 从行人 crop 提取, 与检测时一致
                feature = reid_pipeline.extractor.extract(person_crop)
                feature_vec = json.dumps(feature.tolist())

                # 人脸特征 (512维): 优先在行人 crop 上检测脸 + 5点对齐; 检不到则回退整图
                if fused_pipeline is not None:
                    try:
                        fp = fused_pipeline.face_pipeline
                        for src in (person_crop, img):
                            faces = fp.detector.detect(src)
                            if faces:
                                aligned = align_face(src, faces[0]['keypoints'])
                                face_img = aligned if aligned is not None else faces[0]['crop']
                                face_emb = fp.extractor.extract(face_img)
                                face_vec = json.dumps(face_emb.tolist())
                                break
                    except Exception as e:
                        print(f"[WARN] 人脸特征提取失败(不阻断登记): {e}")

                person_id = db.insert_person(name, id_card, room_num,
                                            feature_vec=feature_vec,
                                            face_vec=face_vec,
                                            face_img_path=face_path)
                # 刷新特征库
                reid_pipeline.feature_db.load_from_mysql()
                if fused_pipeline is not None:
                    fused_pipeline.face_pipeline.face_db.load_from_mysql()

                return jsonify({
                    'success': True, 'message': '登记成功',
                    'person_id': person_id,
                    'feature_extracted': True,
                    'face_extracted': face_vec is not None
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
        if fused_pipeline is not None:
            fused_pipeline.face_pipeline.face_db.load_from_mysql()
    return jsonify({'success': ok})


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="酒店异常人员实时监控识别系统")
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--model', default='saved_models/IBNet_V3_MOT17_Rank1-0987_mAP-0846_ep40.pth')
    parser.add_argument('--yolo_model', default='saved_models/YOLOv8m_V6_HotelDet_mAP50-0822_deploy_ep30.pt')
    parser.add_argument('--video', default=None, help='测试视频路径')
    parser.add_argument('--rtsp', action='store_true', help='使用 RTSP 摄像头')
    parser.add_argument('--db_host', default='localhost')
    parser.add_argument('--db_port', type=int, default=3306)
    parser.add_argument('--db_user', default='root')
    parser.add_argument('--db_password', default='123456')
    parser.add_argument('--db_name', default='hotel_security')

    args = parser.parse_args()

    db_config = {
        'host': args.db_host, 'port': args.db_port,
        'user': args.db_user, 'password': args.db_password,
        'database': args.db_name
    }

    # 初始化系统
    init_system(db_config, args.yolo_model, args.model)

    # 加载已保存的摄像头配置（持久化恢复）
    saved = load_cameras()
    if saved:
        for cam in saved:
            src = cam.get('source')
            if cam.get('source_type') == 'local' and isinstance(src, str) and src.isdigit():
                src = int(src)
            add_camera(src, cam.get('camera_id'), name=cam.get('name'),
                       location=cam.get('location'), source_type=cam.get('source_type'))
    else:
        # 默认: 本地摄像头或测试视频
        if args.video:
            add_camera(args.video, 'CAM_001', name='测试视频', source_type='file')
        elif args.rtsp:
            add_camera('rtsp://localhost:554/stream', 'CAM_001', name='RTSP 摄像头', source_type='rtsp')
        else:
            test_video = PROJECT_ROOT / 'test_video' / 'hotel_raw' / 'demo.mp4'
            if test_video.exists():
                add_camera(str(test_video), 'CAM_001', name='演示视频', source_type='file')
            else:
                add_camera(0, 'CAM_001', name='本地摄像头', source_type='local')

    # 启动检测线程
    det_thread = threading.Thread(target=detection_loop, daemon=True)
    det_thread.start()

    # 启动预览线程（低延迟 MJPEG，独立于检测）
    preview_thread = threading.Thread(target=preview_loop, daemon=True)
    preview_thread.start()

    # 启动摄像头读取 + 检测循环
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