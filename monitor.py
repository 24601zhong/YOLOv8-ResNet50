#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Training Monitor - adapted to current project structure
Usage: python monitor.py
       python monitor.py --watch 60   # auto-refresh every 60s
"""
import os
import sys
import time
import subprocess
import re
import io
from pathlib import Path
from datetime import datetime

# Force UTF-8 on Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(r"c:\D\Myproject\Data-processing\Hotel_Model_Train")

# ============================================================
# Path config
# ============================================================
REID_LOG = PROJECT_ROOT / "resnet50_reid_train/train_output/combined_v2_log/training_log.txt"
REID_BEST = PROJECT_ROOT / "resnet50_reid_train/train_output/combined_v2_log/best_combined_v2.pth"

# V3 training paths
REID_V3_S1_LOG = PROJECT_ROOT / "resnet50_reid_train/train_output/market1501_v3/training_log.txt"
REID_V3_S1_BEST = PROJECT_ROOT / "resnet50_reid_train/train_output/market1501_v3/best_market1501_v3.pth"
REID_V3_S2_LOG = PROJECT_ROOT / "resnet50_reid_train/train_output/mot17_v3/training_log.txt"
REID_V3_S2_BEST = PROJECT_ROOT / "resnet50_reid_train/train_output/mot17_v3/best_mot17_v3.pth"

YOLO_LOG = PROJECT_ROOT / "train_output/yolo_log/hotel_det_v5/training_v5.log"
YOLO_MODEL = PROJECT_ROOT / "yolov8s.pt"
YOLO_MODEL_SIZE = 22_588_772  # expected bytes

# ============================================================
# Helpers
# ============================================================

def run(cmd, timeout=10):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip(), r.stderr.strip(), r.returncode
    except subprocess.TimeoutExpired:
        return "", "timeout", -1
    except Exception as e:
        return "", str(e), -1


def check_gpu():
    out, _, _ = run('nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv,noheader')
    if not out:
        return []
    processes = []
    for line in out.split("\n"):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2:
            processes.append({"pid": parts[0], "name": parts[1], "vram": parts[2] if len(parts) > 2 else "N/A"})
    return processes


def check_python_processes():
    out, _, _ = run('tasklist /FI "IMAGENAME eq python.exe" /FO CSV /NH')
    if not out:
        return []
    procs = []
    for line in out.split("\n"):
        parts = [p.strip('"') for p in line.split(",")]
        if len(parts) >= 2:
            procs.append({"name": parts[0], "pid": parts[1]})
    return procs


def check_reid():
    status = {
        "running": False, "pid": None, "epoch": None,
        "best_rank1": None, "best_epoch": None,
        "loss": None, "micro_f1": None,
        "nan": False, "completed": False, "log_modified": None,
    }

    if not REID_LOG.exists():
        return status

    mtime = datetime.fromtimestamp(os.path.getmtime(REID_LOG))
    status["log_modified"] = mtime

    with open(REID_LOG, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    if "Training Complete" in content:
        status["completed"] = True
    if re.search(r"nan|NaN", content):
        status["nan"] = True

    # Match epoch lines: "  42 |  13.2681  12.9569  0.3112  0.0000 |  0.1734  0.1900 | ..."
    epoch_lines = re.findall(
        r"^\s+(\d+)\s+\|\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+\|\s+([\d.]+)\s+([\d.]+)",
        content, re.MULTILINE
    )
    if epoch_lines:
        last = epoch_lines[-1]
        status["epoch"] = int(last[0])
        status["loss"] = float(last[1])
        status["micro_f1"] = float(last[5])

    best_lines = re.findall(r"\[BEST\] Saved at epoch (\d+) \(MOT17 Rank-1=([\d.]+)\)", content)
    if best_lines:
        last_best = best_lines[-1]
        status["best_rank1"] = float(last_best[1])
        status["best_epoch"] = int(last_best[0])

    return status


def check_reid_v3():
    """Check V3 2-stage ReID training status"""
    status = {
        "stage": None, "running": False, "epoch": None, "total_epochs": None,
        "loss": None, "acc": None, "best_mAP": None, "best_rank1": None,
        "log_modified": None,
    }

    # Determine which stage is active (S2 > S1, newest wins)
    s1_mtime = datetime.fromtimestamp(os.path.getmtime(REID_V3_S1_LOG)) if REID_V3_S1_LOG.exists() else None
    s2_mtime = datetime.fromtimestamp(os.path.getmtime(REID_V3_S2_LOG)) if REID_V3_S2_LOG.exists() else None

    if s2_mtime and (s1_mtime is None or s2_mtime >= s1_mtime):
        status["stage"] = 2
        log_path = REID_V3_S2_LOG
        status["log_modified"] = s2_mtime
    elif s1_mtime:
        status["stage"] = 1
        log_path = REID_V3_S1_LOG
        status["log_modified"] = s1_mtime
    else:
        return status

    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    if "Stage 1 Complete" in content or "Stage 2 Complete" in content:
        status["running"] = False  # completed

    # Parse epoch line: "Epoch  12/80 | Train Loss:..." or "Epoch  12/60 | ..."
    epoch_match = re.findall(r"Epoch\s+(\d+)/(\d+)", content)
    if epoch_match:
        last = epoch_match[-1]
        status["epoch"] = int(last[0])
        status["total_epochs"] = int(last[1])

    # Parse metrics
    mAP_match = re.findall(r"mAP:([\d.]+)", content)
    if mAP_match:
        status["best_mAP"] = float(mAP_match[-1])

    r1_match = re.findall(r"Rank-1:([\d.]+)", content)
    if r1_match:
        status["best_rank1"] = float(r1_match[-1])

    # Parse training metrics from latest line
    loss_match = re.findall(r"Train Loss:([\d.]+)", content)
    if loss_match:
        status["loss"] = float(loss_match[-1])
    acc_match = re.findall(r"Acc:([\d.]+)", content)
    if acc_match:
        status["acc"] = float(acc_match[-1])

    return status


def check_yolo():
    status = {
        "running": False, "pid": None, "epoch": None, "total_epochs": None,
        "mAP50": None, "mAP50_95": None,
        "nan": False, "completed": False, "model_ready": False, "log_modified": None,
    }

    if YOLO_MODEL.exists():
        size = os.path.getsize(YOLO_MODEL)
        if size >= YOLO_MODEL_SIZE * 0.95:
            status["model_ready"] = True

    if not YOLO_LOG.exists():
        return status

    mtime = datetime.fromtimestamp(os.path.getmtime(YOLO_LOG))
    status["log_modified"] = mtime

    with open(YOLO_LOG, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    if re.search(r"nan|NaN", content):
        status["nan"] = True
    if "Training Complete" in content or "Training complete" in content:
        status["completed"] = True

    # YOLO progress: "Epoch 5/60" or "  5/60"
    epoch_match = re.findall(r"Epoch\s+(\d+)/(\d+)", content)
    if not epoch_match:
        epoch_match = re.findall(r"^\s*(\d+)/(\d+)\s", content, re.MULTILINE)
    if epoch_match:
        last = epoch_match[-1]
        status["epoch"] = int(last[0])
        status["total_epochs"] = int(last[1])

    # results.csv
    results_csv = YOLO_LOG.parent / "results.csv"
    if results_csv.exists():
        try:
            with open(results_csv, "r") as f:
                lines = f.readlines()
            if len(lines) > 1:
                parts = lines[-1].strip().split(",")
                if len(parts) > 2:
                    try:
                        status["mAP50"] = float(parts[-2])
                        status["mAP50_95"] = float(parts[-1])
                    except ValueError:
                        pass
        except Exception:
            pass

    return status


# ============================================================
# Report
# ============================================================

def report():
    print(f"\n{'='*60}")
    print(f"  Training Monitor - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    gpu_procs = check_gpu()
    python_procs = check_python_processes()
    python_pids = {p["pid"] for p in python_procs}

    # ---- GPU ----
    print(f"\n  [GPU]")
    if gpu_procs:
        for p in gpu_procs:
            marker = " <-- Python" if p["pid"] in python_pids else ""
            print(f"    PID {p['pid']:>6}  {p['name'][:40]:<40} VRAM={p['vram']}{marker}")
    else:
        print(f"    Idle - no CUDA processes")

    # ---- ReID ----
    reid = check_reid()
    print(f"\n  [ReID Combined V2]")
    print(f"    Log: {REID_LOG}")
    if reid["log_modified"]:
        ago = (datetime.now() - reid["log_modified"]).total_seconds()
        print(f"    Updated: {reid['log_modified'].strftime('%H:%M:%S')} ({ago:.0f}s ago)")
    if reid["completed"]:
        print(f"    [DONE] Training complete!")
    elif reid["epoch"]:
        print(f"    [RUN]  Epoch: {reid['epoch']}/120  |  Loss: {reid['loss']:.4f}  |  MicroF1: {reid['micro_f1']:.4f}")
        if reid["best_rank1"]:
            print(f"    [BEST] Rank-1={reid['best_rank1']:.4f} @ Epoch {reid['best_epoch']}")
        if reid["nan"]:
            print(f"    [WARN] NaN detected!")
    else:
        print(f"    [STOP] Not running")
        if REID_BEST.exists():
            print(f"    Can resume: best_combined_v2.pth")

    # ---- ReID V3 ----
    reid_v3 = check_reid_v3()
    print(f"\n  [ReID V3 (IBN-Net)]")
    if reid_v3["stage"]:
        stage_name = {1: "Market-1501 Pretrain", 2: "MOT17 Finetune"}.get(reid_v3["stage"], "?")
        print(f"    Stage: {reid_v3['stage']} ({stage_name})")
        if reid_v3["epoch"]:
            print(f"    [RUN]  Epoch: {reid_v3['epoch']}/{reid_v3['total_epochs']}  |  "
                  f"Loss: {reid_v3['loss']:.4f}  |  Acc: {reid_v3['acc']:.3f}")
            if reid_v3["best_mAP"]:
                print(f"    mAP: {reid_v3['best_mAP']:.4f}  |  Rank-1: {reid_v3['best_rank1']:.4f}")
        else:
            # Check if stage 1 completed
            s1_done = False
            if REID_V3_S1_LOG.exists():
                with open(REID_V3_S1_LOG, "r", encoding="utf-8", errors="replace") as f:
                    s1_done = "Stage 1 Complete" in f.read()
            if reid_v3["stage"] == 1 and s1_done:
                print(f"    [DONE] Stage 1 complete, ready for Stage 2")
    else:
        print(f"    [STOP] Not started")
        print(f"    Stage 1: {REID_V3_S1_BEST} (pretrained)")
        print(f"    Stage 2: {REID_V3_S2_BEST} (finetuned)")

    # ---- YOLO ----
    yolo = check_yolo()
    print(f"\n  [YOLOv8s V5]")
    print(f"    Log: {YOLO_LOG}")
    if YOLO_MODEL.exists():
        size_kb = os.path.getsize(YOLO_MODEL) // 1024
        pct = os.path.getsize(YOLO_MODEL) / YOLO_MODEL_SIZE * 100
        print(f"    Model: yolov8s.pt ({size_kb}KB / {YOLO_MODEL_SIZE//1024}KB, {pct:.1f}%)")
    else:
        print(f"    Model: not downloaded")

    if yolo["log_modified"]:
        ago = (datetime.now() - yolo["log_modified"]).total_seconds()
        print(f"    Updated: {yolo['log_modified'].strftime('%H:%M:%S')} ({ago:.0f}s ago)")
    if yolo["completed"]:
        print(f"    [DONE] Training complete!")
    elif yolo["epoch"]:
        print(f"    [RUN]  Epoch: {yolo['epoch']}/{yolo.get('total_epochs', '?')}")
        if yolo["mAP50"]:
            print(f"    mAP50: {yolo['mAP50']:.4f}  |  mAP50-95: {yolo['mAP50_95']:.4f}")
        if yolo["nan"]:
            print(f"    [WARN] NaN detected!")
    elif yolo["model_ready"]:
        print(f"    [READY] Model ready, can start training")
    else:
        print(f"    [STOP] yolov8s.pt not ready")

    # ---- Summary ----
    print(f"\n  {'-'*56}")
    anomalies = []
    if reid["nan"]:
        anomalies.append("[WARN] ReID NaN - pause training!")
    if yolo["nan"]:
        anomalies.append("[WARN] YOLO NaN - pause training!")
    if not gpu_procs:
        anomalies.append("[INFO] GPU idle - can start training")
    if reid["completed"]:
        anomalies.append("[DONE] ReID training complete!")
    if yolo["completed"]:
        anomalies.append("[DONE] YOLO training complete!")
    if reid["epoch"] is None and not reid["completed"]:
        anomalies.append("[INFO] ReID not running")
    if yolo["epoch"] is None and not yolo["completed"] and not yolo["model_ready"]:
        anomalies.append("[INFO] YOLO model not downloaded")

    warnings = [a for a in anomalies if a.startswith("[WARN]")]
    if warnings:
        for a in warnings:
            print(f"  {a}")
    if not anomalies:
        print(f"  All systems normal")
    else:
        for a in anomalies:
            print(f"  {a}")
        if not warnings:
            print(f"  No critical warnings")

    print(f"{'='*60}\n")


def main():
    watch = None
    if "--watch" in sys.argv:
        try:
            idx = sys.argv.index("--watch")
            watch = int(sys.argv[idx + 1])
        except (ValueError, IndexError):
            watch = 60

    if watch:
        print(f"Auto-refresh: every {watch}s")
        try:
            while True:
                os.system("cls" if os.name == "nt" else "clear")
                report()
                time.sleep(watch)
        except KeyboardInterrupt:
            print("\nMonitor stopped")
    else:
        report()


if __name__ == "__main__":
    main()
