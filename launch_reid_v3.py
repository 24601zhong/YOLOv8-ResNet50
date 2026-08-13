#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Launch ReID V3 Training Pipeline

Waits for YOLO training to complete, then runs:
  Stage 1: Market-1501 Pretraining (80 epochs, ~2h)
  Stage 2: MOT17 All-Sequence Finetuning (60 epochs, ~2h)
"""
import os
import sys
import time
import subprocess
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(r"c:\D\Myproject\Data-processing\Hotel_Model_Train")
REID_DIR = PROJECT_ROOT / "resnet50_reid_train"
PYTHON = r"C:\D\CondaData\envs_dirs\hotel_det\python.exe"

# Paths
YOLO_LOG = PROJECT_ROOT / "train_output/yolo_log/hotel_det_v5/training_v5.log"
MARKET_DIR = Path(r"c:\D\Myproject\Data-processing\Market-1501-v15.09.15")
PRETRAIN_SCRIPT = REID_DIR / "pretrain_market1501_v3.py"
FINETUNE_SCRIPT = REID_DIR / "finetune_mot17_v3.py"

# Output dirs
S1_OUTPUT = REID_DIR / "train_output/market1501_v3"
S2_OUTPUT = REID_DIR / "train_output/mot17_v3"


def wait_for_yolo_completion(check_interval=30):
    """Wait for YOLO training to finish"""
    print("[launch] Waiting for YOLO training to complete...")
    last_epoch = -1
    stall_count = 0

    while True:
        # Check if log exists
        if not YOLO_LOG.exists():
            print(f"[launch]   YOLO log not found, waiting...")
            time.sleep(check_interval)
            continue

        # Check log content
        with open(YOLO_LOG, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        if "Training Complete" in content or "Training complete" in content:
            print(f"[launch] YOLO training complete!")
            return

        # Check if process is still alive
        import re
        epochs = re.findall(r"(\d+)/60", content)
        if epochs:
            current = int(epochs[-1])
            if current != last_epoch:
                print(f"[launch]   YOLO progress: epoch {current}/60")
                last_epoch = current
                stall_count = 0
            else:
                stall_count += 1

        # If stalled for too long, might be done
        if stall_count > 20:  # 10 min no progress
            print(f"[launch]   YOLO appears stalled (no progress for {stall_count * check_interval}s)")
            # Check if process still running
            import subprocess as sp
            r = sp.run('tasklist /FI "IMAGENAME eq python.exe" /FO CSV /NH',
                      shell=True, capture_output=True, text=True)
            if "python.exe" not in r.stdout:
                print(f"[launch]   No Python process found, assuming YOLO finished")
                return

        time.sleep(check_interval)


def run_stage1():
    """Run Stage 1: Market-1501 Pretraining"""
    print(f"\n{'=' * 70}")
    print(f"  Stage 1: Market-1501 Pretraining (IBNet V3)")
    print(f"  Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 70}")

    S1_OUTPUT.mkdir(parents=True, exist_ok=True)
    log_path = S1_OUTPUT / "training_log.txt"

    cmd = [
        PYTHON, "-u", str(PRETRAIN_SCRIPT),
        "--market_dir", str(MARKET_DIR),
        "--output", str(S1_OUTPUT),
        "--epochs", "80",
        "--batch_p", "16",
        "--batch_k", "4",
        "--lr", "3.5e-4",
        "--warmup", "5",
    ]

    print(f"[launch] Command: {' '.join(cmd)}")

    with open(log_path, "w", encoding="utf-8") as log_f:
        proc = subprocess.Popen(
            cmd,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            cwd=str(REID_DIR),
        )

    print(f"[launch] Stage 1 PID: {proc.pid}")
    proc.wait()

    if proc.returncode != 0:
        print(f"[launch] Stage 1 FAILED with code {proc.returncode}")
        print(f"[launch] Check log: {log_path}")
        return False

    print(f"[launch] Stage 1 complete!")
    print(f"  End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return True


def run_stage2():
    """Run Stage 2: MOT17 Fine-tuning"""
    print(f"\n{'=' * 70}")
    print(f"  Stage 2: MOT17 All-Sequence Fine-tuning (IBNet V3)")
    print(f"  Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 70}")

    S2_OUTPUT.mkdir(parents=True, exist_ok=True)
    log_path = S2_OUTPUT / "training_log.txt"

    pretrained = S1_OUTPUT / "best_market1501_v3.pth"
    if not pretrained.exists():
        print(f"[launch] Stage 1 model not found: {pretrained}")
        print(f"[launch] Looking for checkpoints...")
        ckpts = sorted(S1_OUTPUT.glob("checkpoint_epoch_*.pth"))
        if ckpts:
            pretrained = ckpts[-1]
            print(f"[launch] Using checkpoint: {pretrained}")
        else:
            print(f"[launch] No Stage 1 weights found! Aborting.")
            return False

    cmd = [
        PYTHON, "-u", str(FINETUNE_SCRIPT),
        "--pretrained", str(pretrained),
        "--output", str(S2_OUTPUT),
        "--epochs", "60",
        "--batch_p", "6",
        "--batch_k", "4",
        "--grad_accum", "2",
        "--lr", "1e-4",
        "--early_stop", "15",
        "--eval_every", "10",
    ]

    print(f"[launch] Command: {' '.join(cmd)}")

    with open(log_path, "w", encoding="utf-8") as log_f:
        proc = subprocess.Popen(
            cmd,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            cwd=str(REID_DIR),
        )

    print(f"[launch] Stage 2 PID: {proc.pid}")
    proc.wait()

    if proc.returncode != 0:
        print(f"[launch] Stage 2 FAILED with code {proc.returncode}")
        print(f"[launch] Check log: {log_path}")
        return False

    print(f"[launch] Stage 2 complete!")
    print(f"  End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return True


def main():
    print("=" * 70)
    print("  ReID V3 Training Pipeline")
    print(f"  Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Check paths
    if not MARKET_DIR.exists():
        print(f"[ERROR] Market-1501 not found: {MARKET_DIR}")
        sys.exit(1)

    # Step 0: Wait for YOLO
    wait_for_yolo_completion()

    # Step 1: Stage 1
    if not run_stage1():
        print(f"\n[launch] Pipeline aborted at Stage 1")
        sys.exit(1)

    # Step 2: Stage 2
    if not run_stage2():
        print(f"\n[launch] Pipeline aborted at Stage 2")
        sys.exit(1)

    # Done
    print(f"\n{'=' * 70}")
    print(f"  ReID V3 Pipeline Complete!")
    print(f"  Stage 1 model: {S1_OUTPUT / 'best_market1501_v3.pth'}")
    print(f"  Stage 2 model: {S2_OUTPUT / 'best_mot17_v3.pth'}")
    print(f"  End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
