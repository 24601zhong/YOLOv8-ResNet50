# -*- coding: utf-8 -*-
"""验证 ReID 切换到 IBNetResNet50 V3 后特征提取正确"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'system_server'))

from reid_match import ReidFeatureExtractor

MODEL = ROOT / 'saved_models' / 'IBNet_V3_MOT17_Rank1-0987_mAP-0846_ep40.pth'
IMG_DIR = ROOT / 'dataset' / 'det' / 'hotel_dataset' / 'images' / 'test'


def main():
    t0 = time.time()
    ext = ReidFeatureExtractor(model_path=str(MODEL))
    print(f'[init] model loaded in {time.time()-t0:.1f}s, device={ext.device}')
    print(f'[init] model type: {type(ext.model).__name__}')

    imgs = sorted(IMG_DIR.glob('*.jpg'))[:4]
    for p in imgs:
        frame = cv2.imread(str(p))
        if frame is None:
            continue
        t0 = time.time()
        feat = ext.extract(frame)
        dt = (time.time() - t0) * 1000
        norm = float(np.linalg.norm(feat))
        print(f'  {p.name}: dim={feat.shape[0]}  |L2|={norm:.4f}  finite={np.isfinite(feat).all()}  {dt:.1f}ms')
    return 0


if __name__ == '__main__':
    sys.exit(main())
