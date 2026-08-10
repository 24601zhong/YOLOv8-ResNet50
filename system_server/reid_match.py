# -*- coding: utf-8 -*-
"""
身份相似度匹配核心逻辑 reid_match.py
======================================
完整业务流程（硬编码）：
  1. 实时读取 MySQL 全部在住人员 2048 维特征向量库
  2. ROI 图像输入改进 ResNet50 提取当前行人特征 f_curr
  3. 批量计算 f_curr 与库内所有特征的余弦相似度，取最大相似度 MaxSim
  4. 按阈值 0.85 完成判定：≥0.85 为已登记住客；<0.85 为异常人员
独立调试标准：导入单张行人图片可输出正确相似度数值与身份判定结果
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

import cv2
import torch
import torch.nn.functional as F

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# 特征提取器（集成改进 ResNet50）
# ============================================================

class ReidFeatureExtractor:
    """
    改进 ResNet50 特征提取器
    - 输入：256×128 行人 ROI 图像
    - 输出：2048 维 L2 归一化特征向量
    """

    def __init__(self, model_path=None, device=None):
        """
        :param model_path: 模型权重路径
        :param device: 计算设备
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model = None
        self._load_model(model_path)

        # 图像预处理参数（与训练一致）
        self.img_h = 256
        self.img_w = 128
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        # 统计
        self.extraction_times = []

        print(f"[INFO] ReID 特征提取器初始化完成, 设备: {self.device}")

    def _load_model(self, model_path):
        """加载改进 ResNet50 模型"""
        try:
            sys.path.insert(0, str(PROJECT_ROOT / 'resnet50_reid'))
            from model import ImprovedResNet50ReID

            self.model = ImprovedResNet50ReID(num_classes=1000, feat_dim=2048)
            self.model.to(self.device)
            self.model.eval()

            if model_path and Path(model_path).exists():
                state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
                if 'model_state_dict' in state_dict:
                    state_dict = state_dict['model_state_dict']

                model_dict = self.model.state_dict()
                loaded = 0
                for k, v in state_dict.items():
                    if k in model_dict and v.shape == model_dict[k].shape:
                        model_dict[k] = v
                        loaded += 1

                self.model.load_state_dict(model_dict)
                print(f"[INFO] 加载 ResNet50 权重: {model_path} ({loaded} 参数)")
            else:
                print("[WARN] 未指定模型权重，使用随机初始化（演示模式）")

        except ImportError as e:
            print(f"[WARN] 无法加载改进 ResNet50: {e}")
            self._init_simple_extractor()

    def _init_simple_extractor(self):
        """初始化简单特征提取器（后备方案）"""
        self.model = None
        self.use_hist = True
        print("[INFO] 使用直方图特征提取（后备方案）")

    def extract(self, image):
        """
        提取特征向量
        :param image: BGR 格式图像（任意尺寸）
        :return: 2048 维 numpy 特征向量（L2 归一化）
        """
        start_time = time.time()

        if self.model is not None:
            # 使用深度学习模型
            feature = self._extract_deep(image)
        else:
            # 使用直方图特征（后备）
            feature = self._extract_hist(image)

        # L2 归一化
        feature = feature / (np.linalg.norm(feature) + 1e-8)

        elapsed = (time.time() - start_time) * 1000
        self.extraction_times.append(elapsed)
        if len(self.extraction_times) > 100:
            self.extraction_times = self.extraction_times[-100:]

        return feature.astype(np.float32)

    def _extract_deep(self, image):
        """深度学习特征提取"""
        # 预处理
        img = self._preprocess(image)

        # 推理
        with torch.no_grad():
            feature = self.model.extract_feature(img)

        feature = feature.squeeze(0).cpu().numpy()
        return feature

    def _extract_hist(self, image):
        """直方图特征提取（后备方案，输出 2048 维）"""
        # 提取多通道直方图特征
        features = []

        # BGR 三通道直方图
        for i in range(3):
            hist = cv2.calcHist([image], [i], None, [256], [0, 256])
            features.extend(hist.flatten())

        # 简化到 2048 维
        feature = np.array(features[:2048], dtype=np.float32)
        if len(feature) < 2048:
            feature = np.pad(feature, (0, 2048 - len(feature)))

        return feature

    def _preprocess(self, image):
        """图像预处理"""
        # BGR -> RGB
        if len(image.shape) == 3:
            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            img = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # 调整大小
        img = cv2.resize(img, (self.img_w, self.img_h))

        # 归一化
        img = img.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std

        # HWC -> CHW, 添加 batch 维度
        img = img.transpose(2, 0, 1)
        img_tensor = torch.from_numpy(img).unsqueeze(0).to(self.device)

        return img_tensor

    def get_avg_extraction_time(self):
        """获取平均特征提取耗时"""
        if self.extraction_times:
            return np.mean(self.extraction_times)
        return 0.0


# ============================================================
# 特征库管理器（MySQL 特征存储 + 内存缓存）
# ============================================================

class FeatureDatabase:
    """
    特征数据库管理器
    - 从 MySQL person_info 表加载特征向量
    - 内存缓存 + 增量更新
    - 支持实时刷新
    """

    def __init__(self, db_config=None):
        """
        :param db_config: MySQL 数据库连接配置
        """
        self.db_config = db_config or {
            'host': 'localhost',
            'port': 3306,
            'user': 'root',
            'password': 'root',
            'database': 'hotel_security'
        }

        # 内存特征缓存
        self.feature_ids = []       # 人员ID列表
        self.feature_vectors = None  # numpy 矩阵 [N, 2048]
        self.feature_info = {}      # {person_id: {name, id_card, room_num, ...}}
        self.last_refresh = None
        self.is_loaded = False

    def load_from_mysql(self):
        """从 MySQL 加载所有特征向量"""
        try:
            sys.path.insert(0, str(PROJECT_ROOT / 'system_server'))
            from db_mysql import HotelDatabase

            db = HotelDatabase(**self.db_config)
            if not db.test_connection():
                print("[WARN] MySQL 连接失败，使用示例数据")
                self._load_demo_data()
                return

            # 查询所有在住人员
            persons = db.get_all_persons(limit=10000)

            if not persons:
                print("[INFO] person_info 表为空，使用示例数据")
                self._load_demo_data()
                db.close()
                return

            # 过滤有特征向量的记录
            valid_persons = []
            features = []
            ids = []

            for person in persons:
                if person.get('feature_vec'):
                    try:
                        vec = np.array(json.loads(person['feature_vec']), dtype=np.float32)
                        if vec.shape[0] == 2048:
                            valid_persons.append(person)
                            features.append(vec)
                            ids.append(person['id'])
                    except (json.JSONDecodeError, ValueError):
                        continue

            if not features:
                print("[INFO] 无有效特征向量，使用示例数据")
                self._load_demo_data()
                db.close()
                return

            # 更新缓存
            self.feature_ids = ids
            self.feature_vectors = np.stack(features)
            self.feature_info = {
                p['id']: {
                    'name': p['name'],
                    'id_card': p['id_card'],
                    'room_num': p['room_num'],
                    'check_in_time': str(p.get('check_in_time', '')),
                    'face_img_path': p.get('face_img_path', '')
                }
                for p in valid_persons
            }

            self.last_refresh = datetime.now()
            self.is_loaded = True

            print(f"[INFO] 从 MySQL 加载 {len(features)} 条特征记录")
            db.close()

        except ImportError:
            print("[WARN] 无法导入 db_mysql，使用示例数据")
            self._load_demo_data()
        except Exception as e:
            print(f"[WARN] 数据库加载异常: {e}，使用示例数据")
            self._load_demo_data()

    def _load_demo_data(self):
        """加载示例数据（演示用）"""
        print("[INFO] 生成示例特征库（5 个虚拟人员）")

        self.feature_ids = [1, 2, 3, 4, 5]
        self.feature_vectors = np.random.randn(5, 2048).astype(np.float32)
        self.feature_vectors = self.feature_vectors / np.linalg.norm(self.feature_vectors, axis=1, keepdims=True)

        self.feature_info = {
            1: {'name': '张三', 'id_card': '110101199001011234', 'room_num': '8001',
                'check_in_time': '2026-08-01 14:30:00', 'face_img_path': 'demo/zhangsan.jpg'},
            2: {'name': '李四', 'id_card': '110101198505056789', 'room_num': '8002',
                'check_in_time': '2026-08-02 10:15:00', 'face_img_path': 'demo/lisi.jpg'},
            3: {'name': '王五', 'id_card': '110101199203032345', 'room_num': '8003',
                'check_in_time': '2026-08-03 16:45:00', 'face_img_path': 'demo/wangwu.jpg'},
            4: {'name': '赵六', 'id_card': '110101198811114567', 'room_num': '8004',
                'check_in_time': '2026-08-04 09:20:00', 'face_img_path': 'demo/zhaoliu.jpg'},
            5: {'name': '钱七', 'id_card': '110101199505057890', 'room_num': '8005',
                'check_in_time': '2026-08-05 20:00:00', 'face_img_path': 'demo/qianqi.jpg'}
        }

        self.last_refresh = datetime.now()
        self.is_loaded = True

    def refresh(self):
        """刷新特征库"""
        self.load_from_mysql()

    def get_all_features(self):
        """获取所有特征"""
        return self.feature_vectors, self.feature_ids

    def get_feature_info(self, person_id):
        """获取人员信息"""
        return self.feature_info.get(person_id, {})

    def size(self):
        """获取特征库大小"""
        return len(self.feature_ids)


# ============================================================
# 余弦相似度匹配器
# ============================================================

class CosineSimilarityMatcher:
    """
    余弦相似度匹配器
    - 阈值 0.85：≥0.85 判定为已登记住客
    - <0.85 判定为异常人员
    """

    THRESHOLD = 0.85  # 业务阈值（固定）

    def __init__(self, feature_db, threshold=None):
        """
        :param feature_db: FeatureDatabase 实例
        :param threshold: 相似度阈值（默认 0.85）
        """
        self.feature_db = feature_db
        self.threshold = threshold or self.THRESHOLD

        # 匹配统计
        self.match_stats = {
            'total_queries': 0,
            'matched': 0,
            'unmatched': 0,
            'avg_similarity': 0.0,
            'query_times': []
        }

        print(f"[INFO] 余弦相似度匹配器初始化完成, 阈值: {self.threshold}")

    def match(self, query_feature):
        """
        单次匹配
        :param query_feature: 查询特征向量 [2048]（L2 归一化）
        :return: {
            'is_matched': bool,
            'similarity': float,
            'person_id': int,
            'person_info': dict,
            'is_anomaly': bool
        }
        """
        start_time = time.time()

        self.feature_db.feature_vectors, self.feature_db.feature_ids

        # 如果特征库为空
        if self.feature_db.feature_vectors is None or len(self.feature_db.feature_ids) == 0:
            return {
                'is_matched': False,
                'similarity': 0.0,
                'person_id': -1,
                'person_info': {},
                'is_anomaly': True,
                'message': '特征库为空'
            }

        # 确保查询特征 L2 归一化
        query_norm = query_feature / (np.linalg.norm(query_feature) + 1e-8)

        # 批量计算余弦相似度
        # cos_sim = dot(f_curr, f_i) / (||f_curr|| * ||f_i||)
        # 已 L2 归一化，简化为 dot product
        similarities = np.dot(self.feature_db.feature_vectors, query_norm)

        # 取最大相似度
        max_idx = np.argmax(similarities)
        max_sim = float(similarities[max_idx])
        matched_id = self.feature_db.feature_ids[max_idx]

        # 阈值判定
        is_matched = max_sim >= self.threshold
        is_anomaly = not is_matched

        # 获取人员信息
        person_info = self.feature_db.get_feature_info(matched_id) if is_matched else {}

        # 更新统计
        self.match_stats['total_queries'] += 1
        if is_matched:
            self.match_stats['matched'] += 1
        else:
            self.match_stats['unmatched'] += 1

        # 运行平均相似度
        total = self.match_stats['matched'] + self.match_stats['unmatched']
        prev_avg = self.match_stats['avg_similarity']
        self.match_stats['avg_similarity'] = prev_avg + (max_sim - prev_avg) / total

        query_time = (time.time() - start_time) * 1000
        self.match_stats['query_times'].append(query_time)
        if len(self.match_stats['query_times']) > 100:
            self.match_stats['query_times'] = self.match_stats['query_times'][-100:]

        return {
            'is_matched': is_matched,
            'similarity': max_sim,
            'person_id': matched_id if is_matched else -1,
            'person_info': person_info,
            'is_anomaly': is_anomaly,
            'message': '已登记住客' if is_matched else '异常人员',
            'query_time_ms': query_time
        }

    def batch_match(self, query_features):
        """
        批量匹配
        :param query_features: 查询特征矩阵 [M, 2048]
        :return: 匹配结果列表
        """
        results = []
        for feat in query_features:
            result = self.match(feat)
            results.append(result)
        return results

    def get_stats(self):
        """获取匹配统计"""
        stats = self.match_stats.copy()
        if self.match_stats['query_times']:
            stats['avg_query_time_ms'] = np.mean(self.match_stats['query_times'])
        return stats

    def reset_stats(self):
        """重置统计"""
        self.match_stats = {
            'total_queries': 0,
            'matched': 0,
            'unmatched': 0,
            'avg_similarity': 0.0,
            'query_times': []
        }


# ============================================================
# 完整匹配流水线（提取+匹配+判定）
# ============================================================

class ReidMatchingPipeline:
    """
    完整 ReID 匹配流水线
    - 输入：行人 ROI 图像
    - 输出：身份判定结果
    """

    def __init__(self, model_path=None, db_config=None):
        """
        :param model_path: ResNet50 模型权重路径
        :param db_config: MySQL 数据库配置
        """
        self.extractor = ReidFeatureExtractor(model_path)
        self.feature_db = FeatureDatabase(db_config)
        self.matcher = None

        # 自动加载特征库
        self.feature_db.load_from_mysql()
        self.matcher = CosineSimilarityMatcher(self.feature_db)

        # 流水线统计
        self.pipeline_stats = {
            'total_processed': 0,
            'avg_total_time_ms': 0.0
        }

        print(f"[INFO] ReID 匹配流水线初始化完成")
        print(f"[INFO] 特征库大小: {self.feature_db.size()} 人")

    def process(self, image):
        """
        完整处理流程
        :param image: BGR 格式行人图像
        :return: 匹配结果
        """
        start_time = time.time()

        # Step 1: 特征提取
        feature = self.extractor.extract(image)

        # Step 2: 相似度匹配
        result = self.matcher.match(feature)

        # 计算总耗时
        total_time = (time.time() - start_time) * 1000
        result['total_time_ms'] = total_time

        # 更新统计
        self.pipeline_stats['total_processed'] += 1
        total = self.pipeline_stats['total_processed']
        prev_avg = self.pipeline_stats['avg_total_time_ms']
        self.pipeline_stats['avg_total_time_ms'] = prev_avg + (total_time - prev_avg) / total

        # 附加特征提取时间
        result['extraction_time_ms'] = self.extractor.extraction_times[-1] if self.extractor.extraction_times else 0

        return result

    def process_batch(self, images):
        """批量处理"""
        results = []
        for img in images:
            result = self.process(img)
            results.append(result)
        return results

    def refresh_database(self):
        """刷新特征库"""
        self.feature_db.refresh()
        self.matcher = CosineSimilarityMatcher(self.feature_db)
        print(f"[INFO] 特征库已刷新，当前 {self.feature_db.size()} 人")

    def get_full_stats(self):
        """获取完整统计"""
        return {
            'pipeline': self.pipeline_stats,
            'extractor_avg_time_ms': self.extractor.get_avg_extraction_time(),
            'matcher': self.matcher.get_stats(),
            'feature_db_size': self.feature_db.size()
        }


# ============================================================
# 独立调试入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="ReID 身份匹配调试工具")
    parser.add_argument("--image", type=str,
                        default=None,
                        help="单张行人图片路径（用于调试）")
    parser.add_argument("--model", type=str, default=None,
                        help="ResNet50 模型权重路径")
    parser.add_argument("--threshold", type=float, default=0.85,
                        help="相似度阈值")
    parser.add_argument("--benchmark", action="store_true",
                        help="运行基准测试")
    parser.add_argument("--benchmark_count", type=int, default=100,
                        help="基准测试次数")

    args = parser.parse_args()

    print("=" * 60)
    print("ReID 身份相似度匹配调试工具")
    print("=" * 60)

    # 初始化流水线
    pipeline = ReidMatchingPipeline(model_path=args.model)

    # 单图调试
    if args.image:
        img_path = Path(args.image)
        if img_path.exists():
            img = cv2.imread(str(img_path))
            if img is not None:
                result = pipeline.process(img)
                print(f"\n[单图测试] {args.image}")
                print(f"  判定结果: {'已登记住客' if result['is_matched'] else '异常人员'}")
                print(f"  最大相似度: {result['similarity']:.4f}")
                if result['is_matched']:
                    print(f"  匹配人员: {result['person_info'].get('name', 'N/A')} "
                          f"(房号: {result['person_info'].get('room_num', 'N/A')})")
                print(f"  特征提取耗时: {result.get('extraction_time_ms', 0):.2f}ms")
                print(f"  总处理耗时: {result.get('total_time_ms', 0):.2f}ms")
            else:
                print(f"[ERROR] 图片读取失败: {args.image}")
        else:
            print(f"[ERROR] 图片不存在: {args.image}")
    else:
        print("\n[INFO] 未指定图片，跳过单图测试")

    # 基准测试
    if args.benchmark:
        print(f"\n[INFO] 运行基准测试 ({args.benchmark_count} 次)...")

        # 生成测试图像（随机噪声）
        test_images = [
            np.random.randint(100, 200, (256, 128, 3), dtype=np.uint8)
            for _ in range(min(args.benchmark_count, 10))
        ]

        all_times = []
        for i in range(args.benchmark_count):
            img = test_images[i % len(test_images)]
            result = pipeline.process(img)
            all_times.append(result['total_time_ms'])

            if i < 5 or i % 20 == 0:
                status = "已登记" if result['is_matched'] else "异常"
                print(f"  [{i+1}/{args.benchmark_count}] "
                      f"相似度: {result['similarity']:.4f} | "
                      f"判定: {status} | "
                      f"耗时: {result['total_time_ms']:.2f}ms")

        # 基准统计
        print("\n" + "=" * 60)
        print("基准测试结果")
        print("=" * 60)
        print(f"  测试次数: {len(all_times)}")
        print(f"  平均总耗时: {np.mean(all_times):.2f}ms")
        print(f"  最小耗时: {np.min(all_times):.2f}ms")
        print(f"  最大耗时: {np.max(all_times):.2f}ms")
        print(f"  P95 耗时: {np.percentile(all_times, 95):.2f}ms")

        full_stats = pipeline.get_full_stats()
        print(f"\n  特征提取平均耗时: {full_stats['extractor_avg_time_ms']:.2f}ms")
        print(f"  匹配器查询平均耗时: {full_stats['matcher'].get('avg_query_time_ms', 0):.2f}ms")
        print(f"  总查询次数: {full_stats['matcher']['total_queries']}")
        print(f"  匹配成功: {full_stats['matcher']['matched']}")
        print(f"  匹配失败: {full_stats['matcher']['unmatched']}")

        # 判定准确率
        total = full_stats['matcher']['matched'] + full_stats['matcher']['unmatched']
        if total > 0:
            accuracy = full_stats['matcher']['matched'] / total * 100
            print(f"  基准匹配率: {accuracy:.2f}%")


if __name__ == '__main__':
    main()