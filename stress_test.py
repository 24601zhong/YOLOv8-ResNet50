# -*- coding: utf-8 -*-
"""
系统性能压力测试 stress_test.py
================================
功能：
  1. YOLO 检测器性能基准测试（FPS、推理时间）
  2. ReID 特征提取器性能基准测试
  3. 余弦相似度匹配器批量测试
  4. MJPEG 推流性能测试
  5. 多摄像头并发压力测试
  6. 生成详细性能报告
"""

import os
import sys
import time
import json
import argparse
import statistics
from pathlib import Path
from datetime import datetime

import numpy as np

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


class PerformanceBenchmark:
    """性能基准测试框架"""

    def __init__(self, output_dir='output/benchmark'):
        self.output_dir = PROJECT_ROOT / output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}

    def timed_execution(self, func, iterations=100, warmup=10, label=''):
        """
        计时执行
        :param func: 待测函数
        :param iterations: 迭代次数
        :param warmup: 预热次数
        :param label: 标签
        :return: 统计结果
        """
        # 预热
        for _ in range(warmup):
            func()

        times = []
        for i in range(iterations):
            start = time.perf_counter()
            func()
            elapsed = (time.perf_counter() - start) * 1000  # ms
            times.append(elapsed)

        results = {
            'label': label,
            'iterations': iterations,
            'warmup': warmup,
            'mean_ms': statistics.mean(times),
            'median_ms': statistics.median(times),
            'std_ms': statistics.stdev(times) if len(times) > 1 else 0,
            'min_ms': min(times),
            'max_ms': max(times),
            'p95_ms': sorted(times)[int(len(times) * 0.95)],
            'p99_ms': sorted(times)[int(len(times) * 0.99)],
            'fps': 1000.0 / statistics.mean(times),
            'times': times
        }

        print(f"  {label}:")
        print(f"    平均: {results['mean_ms']:.2f}ms | "
              f"中位: {results['median_ms']:.2f}ms | "
              f"95%: {results['p95_ms']:.2f}ms")
        print(f"    FPS: {results['fps']:.1f}")

        return results

    def benchmark_yolo_detector(self, model_path=None):
        """YOLO 检测器性能测试"""
        print("\n" + "=" * 60)
        print("  YOLO 行人检测器性能测试")
        print("=" * 60)

        try:
            from system_server.video_stream import YOLOPersonDetector

            detector = YOLOPersonDetector(model_path=model_path)

            # 不同分辨率测试
            resolutions = [
                (640, 480),
                (1280, 720),
                (1920, 1080),
            ]

            yolo_results = {}

            for w, h in resolutions:
                dummy_frame = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)

                print(f"\n  分辨率: {w}x{h}")
                result = self.timed_execution(
                    lambda: detector.detect(dummy_frame),
                    iterations=50,
                    warmup=5,
                    label=f"YOLO {w}x{h}"
                )
                yolo_results[f"{w}x{h}"] = result

            self.results['yolo'] = yolo_results

            # 统计
            avg_fps = np.mean([r['fps'] for r in yolo_results.values()])
            print(f"\n  YOLO 平均 FPS: {avg_fps:.1f}")

            return yolo_results

        except ImportError as e:
            print(f"  [SKIP] YOLO 测试跳过: {e}")
            self.results['yolo'] = {'error': str(e)}
            return None

    def benchmark_reid_extractor(self, model_path=None):
        """ReID 特征提取器性能测试"""
        print("\n" + "=" * 60)
        print("  ReID 特征提取器性能测试")
        print("=" * 60)

        try:
            from system_server.reid_match import ReidFeatureExtractor

            extractor = ReidFeatureExtractor(model_path=model_path)

            # 不同尺寸 ROI 测试
            roi_sizes = [
                (256, 128),   # 标准尺寸
                (192, 96),    # 小尺寸
                (320, 160),   # 大尺寸
            ]

            reid_results = {}

            for h, w in roi_sizes:
                dummy_roi = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)

                print(f"\n  ROI 尺寸: {h}x{w}")
                result = self.timed_execution(
                    lambda: extractor.extract(dummy_roi),
                    iterations=100,
                    warmup=10,
                    label=f"ReID 提取 {h}x{w}"
                )
                reid_results[f"{h}x{w}"] = result

            self.results['reid_extract'] = reid_results

            avg_fps = np.mean([r['fps'] for r in reid_results.values()])
            print(f"\n  ReID 提取平均 FPS: {avg_fps:.1f}")

            return reid_results

        except ImportError as e:
            print(f"  [SKIP] ReID 测试跳过: {e}")
            self.results['reid_extract'] = {'error': str(e)}
            return None

    def benchmark_matcher(self, feature_dim=2048, db_size=1000):
        """余弦相似度匹配器性能测试"""
        print("\n" + "=" * 60)
        print(f"  余弦相似度匹配器测试 (库大小: {db_size})")
        print("=" * 60)

        try:
            from system_server.reid_match import FeatureDatabase, CosineSimilarityMatcher

            # 创建模拟特征库
            np.random.seed(42)
            feature_vectors = np.random.randn(db_size, feature_dim).astype(np.float32)
            feature_vectors = feature_vectors / np.linalg.norm(feature_vectors, axis=1, keepdims=True)
            feature_ids = list(range(1, db_size + 1))

            feature_db = FeatureDatabase()
            feature_db.feature_vectors = feature_vectors
            feature_db.feature_ids = feature_ids

            matcher = CosineSimilarityMatcher(feature_db)

            # 查询特征
            query = np.random.randn(feature_dim).astype(np.float32)
            query = query / np.linalg.norm(query)

            # 不同库大小测试
            matcher_results = {}

            for test_size in [10, 50, 100, 500, 1000]:
                sub_db = FeatureDatabase()
                sub_db.feature_vectors = feature_vectors[:test_size]
                sub_db.feature_ids = feature_ids[:test_size]

                sub_matcher = CosineSimilarityMatcher(sub_db)

                print(f"\n  库大小: {test_size}")
                result = self.timed_execution(
                    lambda: sub_matcher.match(query),
                    iterations=200,
                    warmup=20,
                    label=f"Matcher size={test_size}"
                )
                matcher_results[f"size_{test_size}"] = result

            self.results['matcher'] = matcher_results

            return matcher_results

        except ImportError as e:
            print(f"  [SKIP] 匹配器测试跳过: {e}")
            self.results['matcher'] = {'error': str(e)}
            return None

    def benchmark_full_pipeline(self, model_path=None, reid_model_path=None):
        """完整检测流水线性能测试"""
        print("\n" + "=" * 60)
        print("  完整检测流水线性能测试")
        print("=" * 60)

        try:
            from system_server.video_stream import YOLOPersonDetector
            from system_server.reid_match import ReidMatchingPipeline

            detector = YOLOPersonDetector(model_path=model_path)
            pipeline = ReidMatchingPipeline(model_path=reid_model_path)

            # 模拟输入
            dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

            def full_pipeline():
                detections = detector.detect(dummy_frame)
                for det in detections:
                    roi = det.get('roi')
                    if roi is not None and roi.size > 0:
                        pipeline.process(roi)

            pipeline_result = self.timed_execution(
                full_pipeline,
                iterations=30,
                warmup=5,
                label="完整流水线 (YOLO + ReID)"
            )

            self.results['full_pipeline'] = pipeline_result
            return pipeline_result

        except ImportError as e:
            print(f"  [SKIP] 流水线测试跳过: {e}")
            self.results['full_pipeline'] = {'error': str(e)}
            return None

    def benchmark_mjpeg_encoding(self):
        """MJPEG 编码性能测试"""
        print("\n" + "=" * 60)
        print("  MJPEG 编码性能测试")
        print("=" * 60)

        try:
            import cv2

            # 不同分辨率
            resolutions = [
                (640, 480),
                (1280, 720),
                (1920, 1080),
            ]

            mjpeg_results = {}

            for w, h in resolutions:
                dummy_frame = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)

                def encode_frame():
                    return cv2.imencode('.jpg', dummy_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])

                print(f"\n  分辨率: {w}x{h}")
                result = self.timed_execution(
                    encode_frame,
                    iterations=200,
                    warmup=20,
                    label=f"JPEG 编码 {w}x{h}"
                )
                mjpeg_results[f"{w}x{h}"] = result

            self.results['mjpeg'] = mjpeg_results
            return mjpeg_results

        except ImportError as e:
            print(f"  [SKIP] MJPEG 测试跳过: {e}")
            self.results['mjpeg'] = {'error': str(e)}
            return None

    def run_all(self, yolo_model=None, reid_model=None):
        """运行所有基准测试"""
        print("\n" + "#" * 70)
        print("  酒店异常人员监控识别系统 - 性能基准测试")
        print("#" * 70)
        print(f"\n  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  设备: {'CUDA' if self._check_cuda() else 'CPU'}")

        self.benchmark_yolo_detector(yolo_model)
        self.benchmark_reid_extractor(reid_model)
        self.benchmark_matcher()
        self.benchmark_full_pipeline(yolo_model, reid_model)
        self.benchmark_mjpeg_encoding()

        # 生成报告
        self.generate_report()
        return self.results

    def _check_cuda(self):
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def generate_report(self):
        """生成性能报告"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'device': 'CUDA' if self._check_cuda() else 'CPU',
            'results': {}
        }

        # 汇总各模块性能
        for module, data in self.results.items():
            if isinstance(data, dict) and 'error' not in data:
                report['results'][module] = {}
                for config, result in data.items():
                    if isinstance(result, dict) and 'mean_ms' in result:
                        report['results'][module][config] = {
                            'mean_ms': result['mean_ms'],
                            'fps': result['fps'],
                            'p95_ms': result['p95_ms']
                        }

        # 保存
        report_path = self.output_dir / f'benchmark_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print("\n" + "=" * 60)
        print("  性能测试报告已生成")
        print("=" * 60)
        print(f"  报告路径: {report_path}")

        # 输出摘要
        print("\n  性能摘要:")
        print("  " + "-" * 50)

        for module, data in self.results.items():
            if isinstance(data, dict) and 'error' not in data:
                for config, result in data.items():
                    if isinstance(result, dict) and 'fps' in result:
                        print(f"  {module}/{config}: {result['fps']:.1f} FPS "
                              f"({result['mean_ms']:.1f}ms)")

        print("=" * 60)

        return report


def main():
    parser = argparse.ArgumentParser(description="系统性能基准测试")
    parser.add_argument('--yolo-model', default=None, help='YOLO 模型路径')
    parser.add_argument('--reid-model', default=None, help='ReID 模型路径')
    parser.add_argument('--quick', action='store_true', help='快速测试（减少迭代次数）')

    args = parser.parse_args()

    benchmark = PerformanceBenchmark()
    results = benchmark.run_all(args.yolo_model, args.reid_model)

    return 0


if __name__ == '__main__':
    sys.exit(main())