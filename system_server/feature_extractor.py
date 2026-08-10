# -*- coding: utf-8 -*-
"""
特征提取模块 feature_extractor.py
封装改进ResNet50模型的特征提取接口
"""

import os
import sys
import cv2
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'resnet50_reid'))
from model import ImprovedResNet50ReID, CosineSimilarityMatcher


class FeatureExtractor:
    """
    特征提取器
    - 加载改进ResNet50模型
    - 提取2048维行人特征向量
    - 余弦相似度匹配
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

        # 构建模型
        self.model = ImprovedResNet50ReID(num_classes=1000, feat_dim=2048)
        self.model.to(self.device)
        self.model.eval()

        # 加载权重
        if model_path:
            self._load_model(model_path)

        # 图像预处理参数(与训练一致)
        self.img_h = 256
        self.img_w = 128
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        # 特征匹配器
        self.matcher = CosineSimilarityMatcher(threshold=0.85)

        print(f"[INFO] 特征提取器初始化完成, 设备: {self.device}")

    def _load_model(self, model_path):
        """加载模型权重"""
        model_path = Path(model_path)
        if not model_path.exists():
            print(f"[WARN] 模型权重不存在: {model_path}")
            return

        print(f"[INFO] 加载模型权重: {model_path}")
        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)

        # 处理可能的key前缀
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']

        # 加载
        model_dict = self.model.state_dict()
        loaded = 0
        for k, v in state_dict.items():
            if k in model_dict and v.shape == model_dict[k].shape:
                model_dict[k] = v
                loaded += 1

        self.model.load_state_dict(model_dict)
        print(f"[INFO] 成功加载 {loaded} 个参数")

    def extract(self, image):
        """
        从图像提取2048维特征向量
        :param image: BGR格式的OpenCV图像
        :return: 特征向量( numpy array, shape [2048] )
        """
        # 预处理
        img = self._preprocess(image)

        # 提取特征
        with torch.no_grad():
            feature = self.model.extract_feature(img)

        # L2归一化
        feature = feature.squeeze(0).cpu().numpy()
        feature = feature / (np.linalg.norm(feature) + 1e-8)

        return feature

    def extract_from_file(self, image_path):
        """从文件提取特征"""
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"无法读取图片: {image_path}")
        return self.extract(img)

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

        # HWC -> CHW, 添加batch维度
        img = img.transpose(2, 0, 1)
        img_tensor = torch.from_numpy(img).unsqueeze(0).to(self.device)

        return img_tensor

    def compute_similarity(self, feat1, feat2):
        """
        计算两个特征的余弦相似度
        :param feat1: 特征向量 [2048]
        :param feat2: 特征向量 [2048]
        :return: 相似度 (0~1)
        """
        # 确保L2归一化
        feat1 = feat1 / (np.linalg.norm(feat1) + 1e-8)
        feat2 = feat2 / (np.linalg.norm(feat2) + 1e-8)

        # 余弦相似度
        similarity = np.dot(feat1, feat2)
        return float(similarity)

    def match_database(self, query_feature, threshold=0.85):
        """
        在特征库中匹配查询特征
        :param query_feature: 查询特征 [2048]
        :param threshold: 相似度阈值
        :return: (is_matched, similarity, matched_id)
        """
        if self.matcher.feature_db is None:
            raise ValueError("特征数据库未构建")

        query_tensor = torch.from_numpy(query_feature).float()
        is_matched, similarity, matched_idx = self.matcher.match(query_tensor)

        # 检查阈值
        if similarity < threshold:
            return False, similarity, -1

        return True, similarity, matched_idx

    def build_feature_database(self, features_dict):
        """
        构建特征数据库
        :param features_dict: {person_id: feature_vector}
        """
        if not features_dict:
            print("[WARN] 空特征字典")
            return

        ids = list(features_dict.keys())
        features = np.stack([features_dict[pid] for pid in ids])

        features_tensor = torch.from_numpy(features).float()
        labels_tensor = torch.tensor(range(len(ids)))

        self.matcher.build_database(features_tensor, labels_tensor)
        self.id_map = {i: pid for i, pid in enumerate(ids)}

    def get_feature_from_database(self, person_id):
        """从数据库获取特征"""
        if hasattr(self, 'id_map') and person_id in self.id_map.values():
            idx = [k for k, v in self.id_map.items() if v == person_id][0]
            return self.matcher.feature_db[idx].numpy()
        return None