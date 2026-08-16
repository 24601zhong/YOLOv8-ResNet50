"""
============================================================
YOLO 模型评估脚本 (支持 TTA + 分源)
用法:
  python eval_yolo.py <model.pt> [--imgsz 640] [--no-tta]
对以下 6 个验证集各跑一次 val 并汇总 mAP50 / mAP50-95 / P / R:
  部署口径 / 混合口径 / coco / hotel / mot17 / mot20
============================================================
"""
import sys
import os

os.chdir(r'c:\D\Myproject\Data-processing\Hotel_Model_Train')

MODEL = sys.argv[1]
IMGSZ = 640
TTA = True

if '--imgsz' in sys.argv:
    IMGSZ = int(sys.argv[sys.argv.index('--imgsz') + 1])
if '--no-tta' in sys.argv:
    TTA = False

from ultralytics import YOLO

YAMLS = [
    ('deploy', 'dataset/det/hotel_det_v6/hotel_det_v6.yaml'),
    ('mixed ', 'dataset/det/hotel_det_v6/hotel_det_v6_mixed.yaml'),
    ('coco  ', 'dataset/det/hotel_det_v6/hotel_det_v6_src_coco.yaml'),
    ('hotel ', 'dataset/det/hotel_det_v6/hotel_det_v6_src_hotel.yaml'),
    ('mot17 ', 'dataset/det/hotel_det_v6/hotel_det_v6_src_mot17.yaml'),
    ('mot20 ', 'dataset/det/hotel_det_v6/hotel_det_v6_src_mot20.yaml'),
]

model = YOLO(MODEL)

print("=" * 64)
print(f"  YOLO Eval  model={os.path.basename(MODEL)}  imgsz={IMGSZ}  TTA={TTA}")
print("=" * 64)
print(f"  {'split':7s} | {'mAP50':>8s} | {'mAP50-95':>8s} | {'P':>7s} | {'R':>7s}")
print("  " + "-" * 56)

summary = {}
for name, data in YAMLS:
    r = model.val(data=data, imgsz=IMGSZ, batch=8, augment=TTA,
                  plots=False, verbose=False, device=0, workers=0)
    mp = getattr(r.box, 'mp', 0.0)
    mr = getattr(r.box, 'mr', 0.0)
    summary[name.strip()] = (r.box.map50, r.box.map, mp, mr)
    print(f"  {name:7s} | {r.box.map50:8.4f} | {r.box.map:8.4f} | {mp:7.4f} | {mr:7.4f}")

print("=" * 64)
print("  Done.")
