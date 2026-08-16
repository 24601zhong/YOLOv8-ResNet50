# -*- coding: utf-8 -*-
"""
人脸识别匹配核心逻辑 face_match.py
======================================
在 ReID 基础上增加「人脸」身份识别 (跨天认住客, 不依赖衣着/体型):

  链路: 原图裁人 → yolov8n-face 检测人脸 → IResNet50+ArcFace 提取 512 维 →
        与 MySQL person_info.face_vec 批量余弦比对 → 阈值判定

与 reid_match.py 的关系 (融合策略 = 人脸优先, ReID 兜底):
  - 能检出人脸且相似度达标 → 用人脸结果 (跨天稳定)
  - 检不到脸 / 人脸置信不足  → 回退 ReID (全身认体型/衣着)

组件:
  FaceDetector           yolov8n-face 人脸检测
  FaceFeatureExtractor   IResNet50 backbone → 512 维 L2 归一化向量
  FaceFeatureDatabase    MySQL face_vec 特征库 + 内存缓存
  FaceMatchingPipeline   检测 + 提取 + 比对
  FusedMatchingPipeline  人脸优先 + ReID 兜底 融合流水线
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

# 项目根
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
# 让 face_arcface 下的 iresnet / face_embed 可导入
sys.path.insert(0, str(PROJECT_ROOT / 'face_arcface'))

# 默认权重路径
FACE_DET_MODEL = str(PROJECT_ROOT / 'saved_models' / 'yolov8n-face.pt')
FACE_EMB_MODEL = str(PROJECT_ROOT / 'saved_models' / 'IResNet50_ArcFace_CASIAWebFace_LFW-09938_ep20.pth')

# insightface ArcFace 112×112 对齐模板 (左眼/右眼/鼻/左嘴角/右嘴角)
CANONICAL_5PTS = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)


def align_face(image, kpts, out_size=112):
    """
    用 5 点关键点做相似变换, 对齐到 ArcFace 规范 112×112 (等价 insightface norm_crop)
    - image: BGR 图 (可以是整帧/人 crop/脸 crop)
    - kpts: [5,2] 关键点, 坐标须与 image 同坐标系
    - 失败 (关键点缺失/非法/变换失败) 返回 None, 调用方回退原始 crop
    """
    if image is None or kpts is None:
        return None
    kpts = np.asarray(kpts, dtype=np.float32).reshape(-1, 2)
    if kpts.shape[0] != 5 or not np.isfinite(kpts).all():
        return None
    dst = CANONICAL_5PTS.copy()
    try:
        M, _ = cv2.estimateAffinePartial2D(kpts, dst, method=cv2.LMEDS)
    except cv2.error:
        return None
    if M is None:
        return None
    return cv2.warpAffine(image, M, (out_size, out_size))


# ============================================================
# 人脸检测器 (yolov8n-face)
# ============================================================

class FaceDetector:
    """
    yolov8n-face 人脸检测器
    - 输入: BGR 图像 (整帧或人 crop)
    - 输出: 人脸列表 [{bbox, conf, crop}]
    """

    def __init__(self, model_path=None, conf_thres=0.25):
        self.conf_thres = conf_thres
        self.model_path = model_path or FACE_DET_MODEL
        self.model = None
        self._load_model()
        self.inference_times = []

    def _load_model(self):
        try:
            from ultralytics import YOLO
            if Path(self.model_path).exists():
                self.model = YOLO(self.model_path)
                print(f"[INFO] 人脸检测模型加载: {self.model_path}")
            else:
                self.model = YOLO('yolov8n-face.pt')
                print(f"[WARN] 未找到 {self.model_path}, 用默认 yolov8n-face.pt")
        except Exception as e:
            print(f"[WARN] 人脸检测模型加载失败: {e}")
            self.model = None

    def detect(self, image):
        """检测 BGR 图像中的人脸, 返回 [{bbox, conf, crop}]"""
        if self.model is None or image is None or image.size == 0:
            return []

        start = time.time()
        faces = []
        try:
            results = self.model.predict(image, imgsz=640, conf=self.conf_thres, verbose=False)
            boxes = results[0].boxes if len(results) else None
            kpts_obj = results[0].keypoints if len(results) else None
            if boxes is not None:
                h, w = image.shape[:2]
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    x1, y1 = max(0, int(x1)), max(0, int(y1))
                    x2, y2 = min(w, int(x2)), min(h, int(y2))
                    if x2 <= x1 or y2 <= y1:
                        continue
                    crop = image[y1:y2, x1:x2]
                    # 5 点关键点 (原图坐标系), 用于对齐
                    kpts = None
                    if kpts_obj is not None and kpts_obj.xy is not None and len(kpts_obj.xy) > i:
                        kpts = kpts_obj.xy[i].cpu().numpy().astype(np.float32)
                    faces.append({
                        'bbox': [x1, y1, x2, y2],
                        'conf': conf,
                        'crop': crop,
                        'keypoints': kpts,
                    })
        except Exception as e:
            print(f"[WARN] 人脸检测异常: {e}")

        elapsed = (time.time() - start) * 1000
        self.inference_times.append(elapsed)
        if len(self.inference_times) > 100:
            self.inference_times = self.inference_times[-100:]
        return faces

    def get_avg_inference_time(self):
        return np.mean(self.inference_times) if self.inference_times else 0.0


# ============================================================
# 人脸特征提取器 (IResNet50 + ArcFace)
# ============================================================

class FaceFeatureExtractor:
    """
    人脸特征提取器
    - 输入: BGR 人脸 crop (任意尺寸, 内部 resize 到 112×112)
    - 输出: 512 维 L2 归一化向量
    """

    def __init__(self, model_path=None, device=None):
        self.model_path = model_path or FACE_EMB_MODEL
        from face_embed import FaceEmbedder
        self.embedder = FaceEmbedder(self.model_path, device=device)
        self.extraction_times = []
        print(f"[INFO] 人脸特征提取器初始化完成 (512 维)")

    def extract(self, face_bgr):
        """face_bgr: BGR ndarray → 512 维 L2 归一化 np.float32"""
        start = time.time()
        emb = self.embedder.embed(face_bgr)
        elapsed = (time.time() - start) * 1000
        self.extraction_times.append(elapsed)
        if len(self.extraction_times) > 100:
            self.extraction_times = self.extraction_times[-100:]
        return emb.astype(np.float32)

    def get_avg_extraction_time(self):
        return np.mean(self.extraction_times) if self.extraction_times else 0.0


# ============================================================
# 人脸特征库 (MySQL face_vec + 内存缓存)
# ============================================================

class FaceFeatureDatabase:
    """
    人脸特征库管理器
    - 从 MySQL person_info.face_vec 加载 512 维向量
    - 内存缓存 + 增量更新
    """

    def __init__(self, db_config=None):
        self.db_config = db_config or {
            'host': 'localhost',
            'port': 3306,
            'user': 'root',
            'password': '123456',
            'database': 'hotel_security'
        }
        self.feature_ids = []
        self.feature_vectors = None  # [N, 512]
        self.feature_info = {}
        self.last_refresh = None
        self.is_loaded = False

    def load_from_mysql(self):
        try:
            from db_mysql import HotelDatabase
            db = HotelDatabase(**self.db_config)
            if not db.test_connection():
                print("[WARN] MySQL 连接失败, 人脸库为空 (可在登记时实时注册)")
                self.is_loaded = True
                return

            persons = db.get_all_persons(limit=10000)
            if not persons:
                print("[INFO] person_info 表为空")
                db.close()
                self.is_loaded = True
                return

            features, ids, infos = [], [], {}
            for p in persons:
                if p.get('face_vec'):
                    try:
                        vec = np.array(json.loads(p['face_vec']), dtype=np.float32)
                        if vec.shape[0] == 512:
                            features.append(vec)
                            ids.append(p['id'])
                            infos[p['id']] = {
                                'name': p['name'],
                                'id_card': p['id_card'],
                                'room_num': p['room_num'],
                                'check_in_time': str(p.get('check_in_time', '')),
                                'face_img_path': p.get('face_img_path', '')
                            }
                    except (json.JSONDecodeError, ValueError):
                        continue

            self.feature_ids = ids
            self.feature_vectors = np.stack(features) if features else None
            self.feature_info = infos
            self.last_refresh = datetime.now()
            self.is_loaded = True
            print(f"[INFO] 从 MySQL 加载 {len(ids)} 条人脸特征")
            db.close()
        except ImportError:
            print("[WARN] 无法导入 db_mysql, 人脸库为空")
            self.is_loaded = True
        except Exception as e:
            print(f"[WARN] 人脸库加载异常: {e}")
            self.is_loaded = True

    def register(self, person_id, name, face_emb, **meta):
        """运行时注册/更新一个人的人脸特征 (供登记流程用)"""
        if self.feature_vectors is None:
            self.feature_vectors = np.empty((0, 512), dtype=np.float32)
        vec = np.asarray(face_emb, dtype=np.float32).reshape(1, 512)
        if person_id in self.feature_ids:
            idx = self.feature_ids.index(person_id)
            self.feature_vectors[idx] = vec[0]
        else:
            self.feature_ids.append(person_id)
            self.feature_vectors = np.vstack([self.feature_vectors, vec])
        self.feature_info[person_id] = meta or {'name': name}

    def size(self):
        return len(self.feature_ids)

    def get_feature_info(self, person_id):
        return self.feature_info.get(person_id, {})


# ============================================================
# 人脸匹配流水线
# ============================================================

class FaceMatchingPipeline:
    """
    人脸匹配流水线: 检测 → 提取 → 批量余弦比对 → 阈值判定
    """

    def __init__(self, detector=None, extractor=None, face_db=None, threshold=0.4):
        self.detector = detector or FaceDetector()
        self.extractor = extractor or FaceFeatureExtractor()
        self.face_db = face_db or FaceFeatureDatabase()
        self.face_db.load_from_mysql()
        self.threshold = threshold
        print(f"[INFO] 人脸匹配流水线初始化完成, 阈值: {self.threshold}, "
              f"特征库: {self.face_db.size()} 人")

    def _match(self, emb):
        """单个 embedding 与库内所有向量比对, 返回最佳结果"""
        if self.face_db.feature_vectors is None or self.face_db.size() == 0:
            return {'is_matched': False, 'similarity': 0.0, 'person_id': -1,
                    'person_info': {}, 'message': '人脸库为空'}
        sims = np.dot(self.face_db.feature_vectors, emb)  # 已 L2 归一化 → 点积=余弦
        idx = int(np.argmax(sims))
        sim = float(sims[idx])
        pid = self.face_db.feature_ids[idx]
        is_matched = sim >= self.threshold
        return {
            'is_matched': is_matched,
            'similarity': sim,
            'person_id': pid if is_matched else -1,
            'person_info': self.face_db.get_feature_info(pid) if is_matched else {},
            'message': '已登记住客' if is_matched else '异常人员',
        }

    def process(self, image):
        """
        :param image: BGR 图像 (整帧或人 crop)
        :return: {face_detected, is_matched, similarity, person_id, person_info, is_anomaly, message}
        """
        faces = self.detector.detect(image)
        if not faces:
            return {
                'face_detected': False, 'is_matched': False, 'similarity': 0.0,
                'person_id': -1, 'person_info': {}, 'is_anomaly': True,
                'message': '未检测到人脸',
                'feature_vec': None, 'feature_type': 'face',
            }

        best = None
        best_emb = None
        for face in faces:
            # 用 5 点关键点对齐到 ArcFace 规范 112×112, 失败则回退原始 crop
            aligned = align_face(image, face['keypoints'])
            face_img = aligned if aligned is not None else face['crop']
            emb = self.extractor.extract(face_img)
            r = self._match(emb)
            r['face_conf'] = face['conf']
            r['face_bbox'] = face['bbox']
            if best is None or r['similarity'] > best['similarity']:
                best = r
                best_emb = emb

        best['face_detected'] = True
        best['is_anomaly'] = not best['is_matched']
        best['num_faces'] = len(faces)
        best['feature_vec'] = best_emb  # 512 维 L2 归一化向量 (供按人聚类/登记复用)
        best['feature_type'] = 'face'
        return best


# ============================================================
# 融合流水线 (人脸优先 + ReID 兜底)
# ============================================================

class FusedMatchingPipeline:
    """
    人脸优先 + ReID 兜底 融合流水线
    - 输入: 完整 BGR 帧 + 行人 bbox
    - 先对原分辨率人 crop 做人脸识别; 命中则返回人脸结果
    - 未命中 (无脸/低于阈值) 则回退 ReID
    """

    def __init__(self, face_pipeline=None, reid_pipeline=None):
        self.face_pipeline = face_pipeline or FaceMatchingPipeline()
        if reid_pipeline is None:
            from reid_match import ReidMatchingPipeline
            reid_pipeline = ReidMatchingPipeline()
        self.reid_pipeline = reid_pipeline
        print("[INFO] 融合流水线初始化完成 (人脸优先 + ReID 兜底)")

    def process(self, frame, bbox=None):
        """
        :param frame: 完整 BGR 帧
        :param bbox: 行人框 [x1,y1,x2,y2] (None 则整帧)
        :return: 判定结果 (含 matched_by 字段标识用哪路)
        """
        img = self._crop(frame, bbox) if bbox else frame

        # 1) 人脸优先
        face_result = self.face_pipeline.process(img)
        if face_result['is_matched']:
            face_result['matched_by'] = 'face'
            return face_result

        # 2) ReID 兜底 (始终计算 reid 全身特征)
        roi = cv2.resize(img, (128, 384))
        reid_result = self.reid_pipeline.process(roi)
        reid_result['matched_by'] = 'reid'
        reid_result['face_detected'] = face_result['face_detected']

        face_feature_vec = face_result.get('feature_vec')   # 512 维, 检出人脸才非 None
        reid_feature_vec = reid_result.get('feature_vec')   # 2048 维, 始终可用

        # 首选特征向量 (聚类/登记用): 人脸优先, 无人脸才用 ReID
        if face_feature_vec is not None:
            reid_result['feature_vec'] = face_feature_vec
            reid_result['feature_type'] = 'face'

        # 同时携带两路特征, 供 AlertManager 做 face↔reid 跨类型合并 (同一人合并成一条)
        reid_result['face_feature_vec'] = face_feature_vec
        reid_result['reid_feature_vec'] = reid_feature_vec
        return reid_result

    @staticmethod
    def _crop(frame, bbox):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return frame
        return frame[y1:y2, x1:x2]


# ============================================================
# 独立调试入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="人脸识别匹配调试工具")
    parser.add_argument("--image", type=str, default=None, help="单张图片路径 (BGR)")
    parser.add_argument("--face_model", type=str, default=None, help="人脸 embedding 权重路径")
    parser.add_argument("--threshold", type=float, default=0.4, help="人脸相似度阈值")
    args = parser.parse_args()

    print("=" * 60)
    print("人脸识别匹配调试工具")
    print("=" * 60)

    detector = FaceDetector()
    extractor = FaceFeatureExtractor(model_path=args.face_model)
    face_db = FaceFeatureDatabase()
    face_db.load_from_mysql()
    pipeline = FaceMatchingPipeline(detector, extractor, face_db, threshold=args.threshold)

    if args.image and Path(args.image).exists():
        img = cv2.imread(args.image)
        result = pipeline.process(img)
        print(f"\n[单图测试] {args.image}")
        print(f"  检出人脸: {result.get('num_faces', 0)} 张")
        print(f"  判定: {'已登记住客' if result['is_matched'] else '异常/未命中'}")
        print(f"  最大相似度: {result['similarity']:.4f}")
        if result['is_matched']:
            info = result['person_info']
            print(f"  匹配人员: {info.get('name', 'N/A')} (房号 {info.get('room_num', 'N/A')})")
    else:
        print("[INFO] 未指定图片 (--image), 仅初始化。可用于演示模块可正常加载。")


if __name__ == '__main__':
    main()
