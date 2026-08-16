"""
============================================================
yolov8n-face.pt 推理演示 — 在酒店场景图上跑人脸检测
只跑推理看效果 (不计算 mAP)
============================================================
"""
import os
import glob

os.chdir(r'c:\D\Myproject\Data-processing\Hotel_Model_Train')

from ultralytics import YOLO

MODEL = 'saved_models/yolov8n-face.pt'
OUT = 'train_output/face_infer'
os.makedirs(OUT, exist_ok=True)

# 抽 8 张酒店验证集图
imgs = sorted(glob.glob('dataset/det/hotel_dataset/images/val/*.jpg'))[:8]

model = YOLO(MODEL)
print("=" * 60)
print(f"  Face inference  model={MODEL}")
print(f"  images: {len(imgs)}  ->  {OUT}")
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
print(f"  Total faces detected: {total_faces}")
print(f"  Annotated images saved to: {OUT}/vis/")
