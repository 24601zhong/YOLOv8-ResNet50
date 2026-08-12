@echo off
cd /d "c:\D\Myproject\Data-processing\Hotel_Model_Train"
"C:\D\CondaData\envs_dirs\hotel_det\python.exe" -u resnet50_reid_train/train_combined.py
pause
