# -*- coding: utf-8 -*-
"""
真实监控图端到端链路测试
========================
链路: YOLOv8m V6 框人 -> 原分辨率裁人 -> yolov8n-face 框脸 -> 5点对齐 -> ArcFace 提取 embedding
用 dataset/det/hotel_dataset/images/test 真实监控图, 统计:
  - 检出人数 / 人脸数 / 对齐成功率 / embedding 是否正常
  - 各阶段耗时 (评估实际部署可行性)

注: 真实图无身份标注, 只验证「框人→框脸→对齐→提特征」链路跑通与检出率,
    不验证身份匹配 (那部分已由 test_face_match.py 用 CASIA 图覆盖)。
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'system_server'))
sys.path.insert(0, str(ROOT / 'face_arcface'))

from ultralytics import YOLO
from face_match import FaceDetector, FaceFeatureExtractor, align_face

PERSON_MODEL = ROOT / 'saved_models' / 'YOLOv8m_V6_HotelDet_mAP50-0822_deploy_ep30.pt'
IMG_DIR = ROOT / 'dataset' / 'det' / 'hotel_dataset' / 'images' / 'test'


def main(n=12, save_dir=None):
    person_model = YOLO(str(PERSON_MODEL))
    if torch.cuda.is_available():
        person_model.to('cuda')
    face_det = FaceDetector()
    face_ext = FaceFeatureExtractor()

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    imgs = sorted(IMG_DIR.glob('*.jpg'))[:n]
    print(f'[test] {len(imgs)} real surveillance images')

    tot_person = tot_face = tot_align_ok = tot_emb_ok = 0
    t_person = t_face = t_align = t_emb = 0.0
    person_with_face = 0

    for img_path in imgs:
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        # 1) 行人检测
        t0 = time.time()
        res = person_model.predict(frame, imgsz=640, conf=0.3, verbose=False)[0]
        boxes = res.boxes
        t_person += time.time() - t0
        persons = []
        if boxes is not None:
            h, w = frame.shape[:2]
            for b in boxes:
                if int(b.cls[0]) != 0:
                    continue
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                x1, y1, x2, y2 = int(max(0, x1)), int(max(0, y1)), int(min(w, x2)), int(min(h, y2))
                if x2 - x1 < 20 or y2 - y1 < 40:
                    continue
                persons.append([x1, y1, x2, y2])
        tot_person += len(persons)

        # 2) 对每个人 crop 做脸部识别
        for (x1, y1, x2, y2) in persons:
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            t0 = time.time()
            faces = face_det.detect(crop)
            t_face += time.time() - t0
            if not faces:
                continue
            person_with_face += 1
            for face in faces:
                tot_face += 1
                # 3) 对齐 (关键点相对 crop 坐标)
                kpts = face['keypoints']
                t0 = time.time()
                aligned = align_face(crop, kpts)
                t_align += time.time() - t0
                if aligned is not None:
                    tot_align_ok += 1
                else:
                    aligned = face['crop']
                # 4) 提取 embedding
                t0 = time.time()
                emb = face_ext.extract(aligned)
                t_emb += time.time() - t0
                if np.isfinite(emb).all() and abs(float(np.linalg.norm(emb)) - 1.0) < 1e-3:
                    tot_emb_ok += 1

        if save_dir:
            vis = frame.copy()
            for (x1, y1, x2, y2) in persons:
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imwrite(str(save_dir / img_path.name), vis)

    print('=' * 56)
    print('RESULT')
    print('=' * 56)
    print(f'  images processed      : {len(imgs)}')
    print(f'  persons detected      : {tot_person}')
    print(f'  persons w/ face       : {person_with_face}')
    print(f'  faces detected        : {tot_face}')
    print(f'  align success         : {tot_align_ok}/{tot_face}')
    print(f'  embedding valid (|v|=1): {tot_emb_ok}/{tot_face}')
    if tot_person:
        print(f'  face-hit rate (/person): {person_with_face}/{tot_person} = '
              f'{person_with_face / tot_person:.2%}')
    print('-' * 56)
    print(f'  avg person-detect     : {t_person / max(1, len(imgs)) * 1000:.1f} ms/img')
    if tot_face:
        print(f'  avg face-detect       : {t_face / max(1, person_with_face) * 1000:.1f} ms/person')
        print(f'  avg align             : {t_align / tot_face * 1000:.2f} ms/face')
        print(f'  avg face-embed        : {t_emb / tot_face * 1000:.1f} ms/face')
    return 0


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=12)
    ap.add_argument('--save', type=str, default=None)
    a = ap.parse_args()
    sys.exit(main(a.n, a.save))
