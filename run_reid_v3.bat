@echo off
rem Launch ReID V3 two-stage training pipeline (detached from any session)
cd /d C:\D\Myproject\Data-processing\Hotel_Model_Train
C:\D\CondaData\envs_dirs\hotel_det\python.exe -u launch_reid_v3.py > train_output\reid_v3_launch.log 2>&1
