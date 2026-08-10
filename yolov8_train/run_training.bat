@echo off
chcp 65001 >nul
set PYTHONUNBUFFERED=1
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
set OMP_NUM_THREADS=2
set MKL_NUM_THREADS=2

echo ===== YOLOv8 改进模型训练 (速度优化版) =====
echo 时间: %date% %time%
echo 参数: epochs=120 batch=8 workers=0 imgsz=512 patience=12 val_freq=5
echo ★ GPU显存压榨v4.0: batch=8 nbs=64 accumulate=8 有效batch=64
echo ★ VFL向量化+WIoU内联: 冗余clone去除,try/except快速路径
echo ★ 验证频率: 前5轮每轮,之后每5轮(val_freq=5)
echo 注意: workers=0 (Windows spawn 多进程导致 MemoryError)
echo =============================================

cd /d "c:\D\Myproject\Data-processing\Hotel_Model_Train"

"C:\D\CondaData\envs_dirs\hotel_det\python.exe" -u yolov8_train/train_yolo.py --weights yolov8n.pt --epochs 120 --batch 8 --workers 0 --imgsz 512 --patience 12 --val_freq 5 >> train_output\yolo_log\training_console.log 2>&1

echo ===== 训练结束 %date% %time% =====
exit /b %errorlevel%
