# -*- coding: utf-8 -*-
"""
系统集成测试 integration_test.py
================================
功能：
  1. 验证所有模块可正确导入
  2. 验证模块间接口兼容性
  3. 验证 MJPEG 视频推流可行性
  4. 验证 Flask 路由完整性
  5. 生成集成测试报告
"""

import os
import sys
import time
import json
import traceback
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


class IntegrationTester:
    """系统集成测试器"""

    def __init__(self):
        self.results = []
        self.passed = 0
        self.failed = 0
        self.warnings = 0

    def log(self, msg, level='info'):
        timestamp = datetime.now().strftime('%H:%M:%S')
        prefix = {'info': '[INFO]', 'pass': '[PASS]', 'fail': '[FAIL]', 'warn': '[WARN]'}[level]
        color = {'info': '\033[0;37m', 'pass': '\033[0;32m', 'fail': '\033[0;31m', 'warn': '\033[0;33m'}[level]
        reset = '\033[0m'
        print(f"{color}[{timestamp}] {prefix} {msg}{reset}")

    def test(self, name, func):
        """运行单个测试"""
        self.log(f"测试: {name}")
        try:
            result = func()
            if result is None or result is True:
                self.passed += 1
                self.results.append({'name': name, 'status': 'PASS'})
                self.log(f"✓ {name} - 通过", 'pass')
                return True
            else:
                self.failed += 1
                self.results.append({'name': name, 'status': 'FAIL', 'error': str(result)})
                self.log(f"✗ {name} - 失败: {result}", 'fail')
                return False
        except Exception as e:
            self.failed += 1
            self.results.append({'name': name, 'status': 'FAIL', 'error': str(e)})
            self.log(f"✗ {name} - 异常: {e}", 'fail')
            traceback.print_exc()
            return False

    def test_module_imports(self):
        """测试模块导入"""
        self.log("=" * 50)
        self.log("阶段1: 模块导入测试")
        self.log("=" * 50)

        def test_db():
            from system_server.db_mysql import HotelDatabase
            return True

        def test_video():
            from system_server.video_stream import YOLOPersonDetector, MultiCameraManager, CameraStreamReader
            return True

        def test_reid():
            from system_server.reid_match import ReidFeatureExtractor, FeatureDatabase, CosineSimilarityMatcher, ReidMatchingPipeline
            return True

        def test_alert():
            from system_server.alert_manager import AlertManager, AnomalyDetector
            return True

        def test_flask():
            from flask import Flask, render_template, request, jsonify, Response
            from flask_cors import CORS
            return True

        def test_cv2():
            import cv2
            return True

        def test_torch():
            import torch
            self.log(f"  PyTorch版本: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
            return True

        def test_numpy():
            import numpy as np
            return True

        self.test("MySQL 模块", test_db)
        self.test("视频流模块", test_video)
        self.test("ReID 模块", test_reid)
        self.test("预警模块", test_alert)
        self.test("Flask 模块", test_flask)
        self.test("OpenCV", test_cv2)
        self.test("PyTorch", test_torch)
        self.test("NumPy", test_numpy)

    def test_module_interfaces(self):
        """测试模块接口"""
        self.log("=" * 50)
        self.log("阶段2: 模块接口兼容性测试")
        self.log("=" * 50)

        # 测试 ReidMatchingPipeline 接口
        def test_reid_process():
            from system_server.reid_match import ReidMatchingPipeline
            import numpy as np

            pipeline = ReidMatchingPipeline(model_path=None)
            dummy_img = np.random.randint(0, 255, (256, 128, 3), dtype=np.uint8)
            result = pipeline.process(dummy_img)

            required_keys = ['is_matched', 'similarity', 'person_info', 'is_anomaly']
            missing = [k for k in required_keys if k not in result]
            if missing:
                return f"缺少字段: {missing}"
            return True

        # 测试 MultiCameraManager 接口
        def test_camera_manager():
            from system_server.video_stream import MultiCameraManager, YOLOPersonDetector

            detector = YOLOPersonDetector(model_path=None)
            manager = MultiCameraManager(detector, skip_frame=2)

            if not hasattr(manager, 'add_camera'):
                return "缺少 add_camera 方法"
            if not hasattr(manager, 'get_result'):
                return "缺少 get_result 方法"
            if not hasattr(manager, 'start_detection_loop'):
                return "缺少 start_detection_loop 方法"
            if not hasattr(manager, 'stop'):
                return "缺少 stop 方法"
            return True

        # 测试 AlertManager 接口
        def test_alert_manager():
            from system_server.alert_manager import AlertManager
            import numpy as np

            mgr = AlertManager()
            dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

            if not hasattr(mgr, 'trigger_alert'):
                return "缺少 trigger_alert 方法"
            if not hasattr(mgr, 'get_stats'):
                return "缺少 get_stats 方法"

            result = mgr.trigger_alert('TEST_CAM', dummy_frame, 0.5)
            if result is None:
                return "trigger_alert 返回 None"
            return True

        # 测试 db_mysql 接口
        def test_db_interface():
            from system_server.db_mysql import HotelDatabase

            db = HotelDatabase(host='localhost', port=3306, user='root', password='123456', database='hotel_security')

            required_methods = ['get_statistics', 'get_all_alerts', 'search_persons',
                                'insert_person', 'insert_alert', 'update_alert_status',
                                'delete_alert', 'update_person', 'delete_person',
                                'get_person_by_id_card', 'test_connection']
            missing = [m for m in required_methods if not hasattr(db, m)]
            if missing:
                return f"缺少方法: {missing}"
            return True

        # 测试 YOLOPersonDetector 接口
        def test_yolo_detector():
            from system_server.video_stream import YOLOPersonDetector
            import numpy as np

            detector = YOLOPersonDetector(model_path=None)
            dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

            if not hasattr(detector, 'detect'):
                return "缺少 detect 方法"

            detections = detector.detect(dummy_frame)
            if not isinstance(detections, list):
                return f"detect 返回类型错误: {type(detections)}"

            # 检查检测结果格式
            for det in detections:
                required_keys = ['bbox', 'confidence']
                missing = [k for k in required_keys if k not in det]
                if missing:
                    return f"检测结果缺少字段: {missing}"

            return True

        self.test("ReID process() 接口", test_reid_process)
        self.test("CameraManager 接口", test_camera_manager)
        self.test("AlertManager 接口", test_alert_manager)
        self.test("MySQL 接口", test_db_interface)
        self.test("YOLO 检测器接口", test_yolo_detector)

    def test_flask_routes(self):
        """测试 Flask 路由"""
        self.log("=" * 50)
        self.log("阶段3: Flask 路由测试")
        self.log("=" * 50)

        def test_dashboard_route():
            from system_server.app import app
            with app.test_client() as client:
                resp = client.get('/')
                if resp.status_code != 200:
                    return f"状态码: {resp.status_code}"
            return True

        def test_monitor_route():
            from system_server.app import app
            with app.test_client() as client:
                resp = client.get('/monitor')
                if resp.status_code != 200:
                    return f"状态码: {resp.status_code}"
            return True

        def test_register_route():
            from system_server.app import app
            with app.test_client() as client:
                resp = client.get('/register')
                if resp.status_code != 200:
                    return f"状态码: {resp.status_code}"
            return True

        def test_alerts_route():
            from system_server.app import app
            with app.test_client() as client:
                resp = client.get('/alerts')
                if resp.status_code != 200:
                    return f"状态码: {resp.status_code}"
            return True

        def test_persons_route():
            from system_server.app import app
            with app.test_client() as client:
                resp = client.get('/persons')
                if resp.status_code != 200:
                    return f"状态码: {resp.status_code}"
            return True

        def test_api_stats():
            from system_server.app import app
            with app.test_client() as client:
                resp = client.get('/api/stats')
                if resp.status_code != 200:
                    return f"状态码: {resp.status_code}"
                data = resp.get_json()
                if 'system' not in data:
                    return "缺少 system 字段"
            return True

        def test_api_cameras():
            from system_server.app import app
            with app.test_client() as client:
                resp = client.get('/api/cameras')
                if resp.status_code != 200:
                    return f"状态码: {resp.status_code}"
                data = resp.get_json()
                if not isinstance(data, list):
                    return "返回类型错误"
            return True

        def test_mjpeg_stream():
            from system_server.app import app
            with app.test_client() as client:
                resp = client.get('/video_feed/CAM_001')
                if resp.status_code != 200:
                    return f"状态码: {resp.status_code}"
                if 'multipart/x-mixed-replace' not in resp.content_type:
                    return f"内容类型错误: {resp.content_type}"
            return True

        self.test("仪表板路由 /", test_dashboard_route)
        self.test("监控路由 /monitor", test_monitor_route)
        self.test("登记路由 /register", test_register_route)
        self.test("预警路由 /alerts", test_alerts_route)
        self.test("人员路由 /persons", test_persons_route)
        self.test("API 统计 /api/stats", test_api_stats)
        self.test("API 摄像头 /api/cameras", test_api_cameras)
        self.test("MJPEG 流 /video_feed", test_mjpeg_stream)

    def test_file_structure(self):
        """测试文件结构"""
        self.log("=" * 50)
        self.log("阶段4: 文件结构完整性测试")
        self.log("=" * 50)

        def check_file(path, desc):
            p = Path(path)
            if not p.exists():
                return f"文件不存在: {path}"
            if p.stat().st_size == 0:
                return f"文件为空: {path}"
            return True

        self.test("Flask 主应用", lambda: check_file(
            PROJECT_ROOT / 'system_server' / 'app.py', 'Flask主应用'))

        self.test("视频流模块", lambda: check_file(
            PROJECT_ROOT / 'system_server' / 'video_stream.py', '视频流模块'))

        self.test("ReID 匹配模块", lambda: check_file(
            PROJECT_ROOT / 'system_server' / 'reid_match.py', 'ReID模块'))

        self.test("预警模块", lambda: check_file(
            PROJECT_ROOT / 'system_server' / 'alert_manager.py', '预警模块'))

        self.test("数据库模块", lambda: check_file(
            PROJECT_ROOT / 'system_server' / 'db_mysql.py', '数据库模块'))

        self.test("MJPEG 服务", lambda: check_file(
            PROJECT_ROOT / 'system_server' / 'mjpeg_server.py', 'MJPEG服务'))

        self.test("监控页面", lambda: check_file(
            PROJECT_ROOT / 'system_server' / 'templates' / 'monitor_v2.html', '监控页面'))

        self.test("仪表板页面", lambda: check_file(
            PROJECT_ROOT / 'system_server' / 'templates' / 'dashboard.html', '仪表板'))

        self.test("登记页面", lambda: check_file(
            PROJECT_ROOT / 'system_server' / 'templates' / 'register.html', '登记页面'))

    def test_integration_flow(self):
        """测试集成流程"""
        self.log("=" * 50)
        self.log("阶段5: 核心流程集成测试")
        self.log("=" * 50)

        def test_detection_pipeline():
            """测试完整检测流水线"""
            import numpy as np
            from system_server.video_stream import YOLOPersonDetector
            from system_server.reid_match import ReidMatchingPipeline
            from system_server.alert_manager import AlertManager

            # 创建组件
            detector = YOLOPersonDetector(model_path=None)
            pipeline = ReidMatchingPipeline(model_path=None)
            alert_mgr = AlertManager()

            # 模拟输入
            dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

            # Step 1: YOLO 检测
            detections = detector.detect(dummy_frame)
            self.log(f"  YOLO 检测到 {len(detections)} 个目标")

            # Step 2: ReID 匹配
            for det in detections:
                roi = det.get('roi')
                if roi is not None and roi.size > 0:
                    match_result = pipeline.process(roi)
                    self.log(f"  ReID 匹配: matched={match_result['is_matched']}, "
                             f"similarity={match_result['similarity']:.3f}")

                    # Step 3: 异常检测
                    if match_result['is_anomaly']:
                        alert_mgr.trigger_alert('TEST', dummy_frame, match_result['similarity'])
                        self.log(f"  预警触发!")

            return True

        def test_mjpeg_protocol():
            """测试 MJPEG 协议"""
            import numpy as np
            import cv2
            import time

            # 模拟 MJPEG 帧编码
            dummy_frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)

            start = time.time()
            for _ in range(100):
                _, jpeg = cv2.imencode('.jpg', dummy_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            elapsed = (time.time() - start) * 1000
            avg_time = elapsed / 100

            self.log(f"  JPEG 编码平均耗时: {avg_time:.1f}ms")

            if avg_time > 50:
                return f"编码耗时过高: {avg_time:.1f}ms/帧"
            return True

        self.test("检测流水线集成", test_detection_pipeline)
        self.test("MJPEG 协议性能", test_mjpeg_protocol)

    def run_all(self):
        """运行所有测试"""
        self.log("\n" + "=" * 60)
        self.log("  酒店异常人员实时监控识别系统 - 集成测试")
        self.log("=" * 60 + "\n")

        try:
            self.test_module_imports()
            self.test_module_interfaces()
            self.test_flask_routes()
            self.test_file_structure()
            self.test_integration_flow()
        except Exception as e:
            self.log(f"测试过程异常: {e}", 'fail')
            traceback.print_exc()

        # 输出报告
        self.log("\n" + "=" * 60)
        self.log("  集成测试报告")
        self.log("=" * 60)

        total = self.passed + self.failed
        self.log(f"总测试数: {total}")
        self.log(f"通过: {self.passed}")
        self.log(f"失败: {self.failed}")
        self.log(f"通过率: {self.passed / total * 100:.1f}%")

        if self.failed > 0:
            self.log("\n失败详情:", 'fail')
            for r in self.results:
                if r['status'] == 'FAIL':
                    self.log(f"  - {r['name']}: {r.get('error', '未知错误')}", 'fail')

        self.log("=" * 60)

        # 保存报告
        report = {
            'timestamp': datetime.now().isoformat(),
            'total': total,
            'passed': self.passed,
            'failed': self.failed,
            'pass_rate': self.passed / total * 100 if total > 0 else 0,
            'results': self.results
        }

        report_path = PROJECT_ROOT / 'output' / 'integration_report.json'
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        self.log(f"报告已保存: {report_path}")

        return self.failed == 0


def main():
    tester = IntegrationTester()
    success = tester.run_all()
    return 0 if success else 1


if __name__ == '__main__':
    import sys
    sys.exit(main())