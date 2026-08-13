@echo off
cd /d "c:\D\Myproject\Data-processing\Hotel_Model_Train"

REM 检查是否从 checkpoint 续训
if "%~1"=="" (
    echo Starting new training...
    "C:\D\CondaData\envs_dirs\hotel_det\python.exe" -u resnet50_reid_train/train_combined.py
) else (
    echo Resuming from checkpoint: %~1
    "C:\D\CondaData\envs_dirs\hotel_det\python.exe" -u resnet50_reid_train/train_combined.py --resume %~1
)
pause
