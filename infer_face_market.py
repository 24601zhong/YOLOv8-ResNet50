"""
============================================================
yolov8n-face.pt 推理演示 — Market-1501 行人图人脸检测
只跑推理看效果 (不计算 mAP)
============================================================
"""
import os
import glob
import random

os.chdir(r'c:\D\Myproject\Data-processing\Hotel_Model_Train')

from ultralytics import YOLO

MODEL = 'saved_models/yolov8n-face.pt'
MARKET = r'c:\D\Myproject\Data-processing\Market-1501-v15.09.15\bounding_box_train'
OUT = 'train_output/face_infer_market'
os.makedirs(OUT, exist_ok=True)

# 随机抽 12 张行人图 (固定种子可复现)
all_imgs = sorted(glob.glob(os.path.join(MARKET, '*.jpg')))
random.seed(42)
imgs = random.sample(all_imgs, 12)

model = YOLO(MODEL)
print("=" * 60)
print(f"  Face inference  model={MODEL}")
print(f"  Market-1501 bounding_box_train: {len(all_imgs)} 张, 抽 {len(imgs)} 张")
print("=" * 60)

total_faces = 0
for p in imgs:
    r = model.predict(p, imgsz=640, conf=0.25, save=True,
                      project=OUT, name='vis', exist_ok=True,
                      verbose=False)
    n = len(r[0].boxes) if r[0].boxes is not None else 0
    total_faces += n
    confs = [round(float(c), 2) for c in r[0].boxes.conf] if n else []
    print(f"  {os.path.basename(p):20s}  faces={n:2d}  conf={confs}")

print("=" * 60)
print(f"  Total faces detected: {total_faces} / {len(imgs)} images")
print(f"  Annotated images saved to: {OUT}/vis/")
