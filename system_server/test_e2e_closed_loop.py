# -*- coding: utf-8 -*-
"""
完整闭环测试: 登记 -> 检测 -> 融合匹配
======================================
1. init_system 真实加载 (MySQL + IBNet V3 ReID + yolov8n-face + ArcFace)
2. 用 CASIA 身份 A 的图1 登记: 提取 face_vec(512) + feature_vec(2048) 写库
3. 用身份 A 的图2 (跨图) 检测: 应 face-first 命中 -> matched_by='face'
4. 用未登记身份 B 的图检测: 应拒绝
"""
import sys
import json
from pathlib import Path

import cv2

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'system_server'))
sys.path.insert(0, str(ROOT / 'face_arcface'))

from face_match import align_face
import app

DB = {'host': 'localhost', 'port': 3306, 'user': 'root',
      'password': '123456', 'database': 'hotel_security'}
REID_MODEL = str(ROOT / 'saved_models' / 'IBNet_V3_MOT17_Rank1-0987_mAP-0846_ep40.pth')
CASIA = ROOT / 'dataset' / 'face_casia' / 'images'


def list_identities(n=3):
    ids = []
    for d in sorted(CASIA.iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else 0):
        if d.is_dir():
            imgs = sorted(d.glob('*.jpg'))
            if len(imgs) >= 2:
                ids.append((d, imgs))
        if len(ids) >= n:
            break
    return ids


def main():
    app.init_system(DB, yolo_model_path=None, reid_model_path=REID_MODEL)
    fp = app.fused_pipeline.face_pipeline

    ids = list_identities(3)
    if len(ids) < 3:
        print('[FAIL] CASIA 身份不足')
        return 1

    # ---- 登记身份 A ----
    dA, imgsA = ids[0]
    label = dA.name
    enroll = cv2.imread(str(imgsA[0]))
    query = cv2.imread(str(imgsA[1]))

    faces = fp.detector.detect(enroll)
    if not faces:
        print('[FAIL] 登记图未检出人脸')
        return 1
    aligned = align_face(enroll, faces[0]['keypoints'])
    face_img = aligned if aligned is not None else faces[0]['crop']
    face_emb = fp.extractor.extract(face_img)
    feature = app.reid_pipeline.extractor.extract(enroll)

    person_id = app.db.insert_person(
        name=f'住客{label}', id_card=f'TEST{label}', room_num='9001',
        feature_vec=json.dumps(feature.tolist()),
        face_vec=json.dumps(face_emb.tolist()))
    app.reid_pipeline.feature_db.load_from_mysql()
    fp.face_db.load_from_mysql()
    print(f'\n[登记] label={label} person_id={person_id} '
          f'face_dim={len(face_emb)} reid_dim={len(feature)}')
    print(f'[库] face_db={fp.face_db.size()} 人  reid_db={app.reid_pipeline.feature_db.size()} 人')

    # ---- 检测身份 A 跨图 (face-first 应命中) ----
    r = app.fused_pipeline.process(query)
    ok = r['is_matched'] and r['person_id'] == person_id and r['matched_by'] == 'face'
    print(f'\n[检测1] 同人跨图 (label {label} 图2):')
    print(f'  {"[OK]" if ok else "[FAIL]"} 命中={r["is_matched"]} person_id={r["person_id"]} '
          f'sim={r["similarity"]:.4f} matched_by={r["matched_by"]}')

    # ---- 检测身份 B (未登记, 应拒绝) ----
    dB, imgsB = ids[1]
    stranger = cv2.imread(str(imgsB[1]))
    r2 = app.fused_pipeline.process(stranger)
    ok2 = not r2['is_matched']
    print(f'\n[检测2] 陌生人 (label {dB.name}, 未登记):')
    print(f'  {"[OK] 正确拒绝" if ok2 else "[FAIL] 误报"} 命中={r2["is_matched"]} '
          f'sim={r2["similarity"]:.4f} matched_by={r2["matched_by"]}')

    print('\n=== 闭环结果 ===')
    print(f'  face-first 跨图命中 : {"PASS" if ok else "FAIL"}')
    print(f'  陌生人拒绝          : {"PASS" if ok2 else "FAIL"}')
    return 0 if (ok and ok2) else 1


if __name__ == '__main__':
    sys.exit(main())
