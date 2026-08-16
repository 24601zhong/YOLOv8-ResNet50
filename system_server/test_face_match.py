# -*- coding: utf-8 -*-
"""
端到端测试: 人脸识别链路 (face_match.py)
========================================
构建演示人脸库 (从 dataset/face_casia/images 取前 N 个身份各注册 1 张),
再验证:
  1) 同身份跨图 (注册图 vs 另一张图) 应高相似度 → 命中
  2) 不同身份 → 低相似度 → 拒绝
  3) FusedMatchingPipeline 能正常构造 (人脸优先 + ReID 兜底)
"""
import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'system_server'))
sys.path.insert(0, str(PROJECT_ROOT / 'face_arcface'))

import cv2
import numpy as np

from face_match import (
    FaceDetector, FaceFeatureExtractor, FaceFeatureDatabase,
    FaceMatchingPipeline, FusedMatchingPipeline, align_face,
)

DATASET = PROJECT_ROOT / 'dataset' / 'face_casia' / 'images'


def list_identities(n=3):
    """返回前 n 个身份目录, 每个目录里至少 2 张图"""
    ids = []
    for d in sorted(DATASET.iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else 0):
        if d.is_dir():
            imgs = sorted(d.glob('*.jpg'))
            if len(imgs) >= 2:
                ids.append(d)
        if len(ids) >= n:
            break
    return ids


def main():
    print('=' * 60)
    print('人脸识别链路端到端测试')
    print('=' * 60)

    # --- 构造组件 (不连 MySQL, 用内存 demo 库) ---
    detector = FaceDetector()
    extractor = FaceFeatureExtractor()
    face_db = FaceFeatureDatabase()

    # 取前 3 个身份, 各注册第 1 张图
    id_dirs = list_identities(3)
    if len(id_dirs) < 2:
        print(f'[ERROR] 数据不足, 仅找到 {len(id_dirs)} 个身份')
        return 1

    print(f'\n[注册] 从 {DATASET} 取 {len(id_dirs)} 个身份, 各注册 1 张图:')
    enroll_map = {}   # label -> (register_img_path, query_img_path)
    for d in id_dirs:
        imgs = sorted(d.glob('*.jpg'))
        enroll_img = cv2.imread(str(imgs[0]))
        faces = detector.detect(enroll_img)
        if not faces:
            print(f'  [WARN] 身份 {d.name} 注册图未检出人脸, 跳过')
            continue
        aligned = align_face(enroll_img, faces[0]['keypoints'])
        face_img = aligned if aligned is not None else faces[0]['crop']
        emb = extractor.extract(face_img)
        face_db.register(person_id=int(d.name), name=f'住客{d.name}', face_emb=emb)
        # 选一张能检出人脸的图当查询图 (跨图, 非注册图)
        query_p = None
        for p in imgs[1:]:
            if detector.detect(cv2.imread(str(p))):
                query_p = p
                break
        if query_p is None:
            query_p = imgs[1]
        enroll_map[d.name] = (imgs[0], query_p)
        print(f'  身份 {d.name}: 注册图 {imgs[0].name}, 查询图 {query_p.name} ({len(imgs)} 张图)')

    print(f'\n[库] 共 {face_db.size()} 人')
    if face_db.size() < 2:
        print('[ERROR] 有效注册人数不足')
        return 1

    # 阈值: 跨天/跨图场景 0.4 偏严, 演示用 0.35 便于观察
    pipeline = FaceMatchingPipeline(detector, extractor, face_db, threshold=0.35)

    print('\n[测试] 同身份跨图 (应命中):')
    for label, (enroll_p, query_p) in enroll_map.items():
        if label not in [str(feature_id) for feature_id in face_db.feature_ids]:
            continue
        query_img = cv2.imread(str(query_p))
        r = pipeline.process(query_img)
        flag = '[OK]' if r['is_matched'] else '[FAIL]'
        print(f'  {flag} 身份 {label} 查询图 {Path(query_p).name}: '
              f"sim={r['similarity']:.4f} 匹配ID={r['person_id']} 命中={r['is_matched']}")

    print('\n[测试] 跨身份 / 陌生人 (应拒绝):')
    # 取一个未注册的身份当陌生人
    stranger_dir = None
    for d in sorted(DATASET.iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else 0):
        if d.is_dir() and d.name not in enroll_map:
            stranger_dir = d
            break
    if stranger_dir:
        simg = sorted(stranger_dir.glob('*.jpg'))[0]
        r = pipeline.process(cv2.imread(str(simg)))
        flag = '[OK] 正确拒绝' if not r['is_matched'] else '[FAIL] 误报'
        print(f'  {flag} 陌生人 {stranger_dir.name} 图 {Path(simg).name}: '
              f"sim={r['similarity']:.4f} 命中={r['is_matched']}")
    # 已注册身份 A 的图, 不应命中身份 B
    labels = list(enroll_map.keys())
    if len(labels) >= 2:
        a = labels[0]
        r = pipeline.process(cv2.imread(str(enroll_map[a][1])))
        print(f'  身份 {a} 图 -> 命中 ID={r["person_id"]} sim={r["similarity"]:.4f} '
              f"({'[OK] 命中自己' if r['is_matched'] and r['person_id'] == int(a) else '[FAIL] 异常'})")

    # --- 融合流水线构造冒烟 ---
    print('\n[测试] FusedMatchingPipeline 构造 (人脸优先 + ReID 兜底):')
    try:
        fused = FusedMatchingPipeline(face_pipeline=pipeline)
        print('  [OK] 融合流水线构造成功')
    except Exception as e:
        print(f'  [WARN] 融合流水线构造失败 (可能缺 ReID 依赖): {e}')

    print('\n=' * 60)
    print('端到端测试完成')
    print('=' * 60)
    return 0


if __name__ == '__main__':
    sys.exit(main())
