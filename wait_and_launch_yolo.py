#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Wait for yolov8s.pt download to complete, then launch YOLOv8s V5 training.
"""
import os
import sys
import time
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(r"c:\D\Myproject\Data-processing\Hotel_Model_Train")
MODEL_PATH = PROJECT_ROOT / "yolov8s.pt"
EXPECTED_SIZE = 22_588_772  # ~22.5MB
PYTHON = r"C:\D\CondaData\envs_dirs\hotel_det\python.exe"
TRAIN_SCRIPT = PROJECT_ROOT / "train_output/yolo_log/hotel_det_v5/train_v5_improved.py"
LOG_FILE = PROJECT_ROOT / "train_output/yolo_log/hotel_det_v5/training_v5.log"

def main():
    print("[wait-yolo] Waiting for yolov8s.pt download to complete...")
    print(f"[wait-yolo] Expected size: {EXPECTED_SIZE:,} bytes ({EXPECTED_SIZE/1024/1024:.1f} MB)")

    last_size = -1
    stall_count = 0

    while True:
        if MODEL_PATH.exists():
            size = os.path.getsize(MODEL_PATH)
            pct = size / EXPECTED_SIZE * 100
            print(f"[wait-yolo] Download progress: {size:,} / {EXPECTED_SIZE:,} bytes ({pct:.1f}%)")

            if size >= EXPECTED_SIZE * 0.99:
                print("[wait-yolo] Download complete! Launching YOLOv8s V5 training...")

                # Clear old log
                if LOG_FILE.exists():
                    os.remove(LOG_FILE)

                # Launch training
                os.chdir(PROJECT_ROOT)
                proc = subprocess.Popen(
                    [PYTHON, "-u", str(TRAIN_SCRIPT)],
                    stdout=open(LOG_FILE, "w"),
                    stderr=subprocess.STDOUT,
                )
                print(f"[wait-yolo] Training launched! PID={proc.pid}")
                print(f"[wait-yolo] Monitor with: python monitor.py")
                return

            if size == last_size:
                stall_count += 1
                if stall_count > 12:  # 2 minutes stalled
                    print("[wait-yolo] Download stalled, waiting...")
                    stall_count = 0
            else:
                stall_count = 0
            last_size = size
        else:
            print("[wait-yolo] Model file not yet created...")

        time.sleep(10)

if __name__ == "__main__":
    main()
