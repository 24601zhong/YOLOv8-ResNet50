# -*- coding: utf-8 -*-
"""app.py 启动冒烟: 只跑 init_system, 验证整条初始化链路不崩"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'system_server'))

t0 = time.time()
import app

db_config = {'host': 'localhost', 'port': 3306, 'user': 'root',
             'password': '123456', 'database': 'hotel_security'}
app.init_system(db_config, yolo_model_path=None, reid_model_path=None)
dt = time.time() - t0

print('\n=== BOOT SMOKE RESULT ===')
print(f'init_system completed in {dt:.1f}s')
print('yolo_detector :', app.yolo_detector is not None, '| device:', app.yolo_detector.device)
print('reid_pipeline :', app.reid_pipeline is not None, '| feature_db size:', app.reid_pipeline.feature_db.size())
print('fused_pipeline:', app.fused_pipeline is not None)
print('  face_db size :', app.fused_pipeline.face_pipeline.face_db.size())
print('alert_mgr     :', app.alert_mgr is not None)
print('system_running:', app.system_running)
print('=== BOOT OK ===')
