"""
============================================================
MOT17 全序列 ReID 数据清洗: mot17_reid_all_sequences.py
============================================================
V3 改进:
  1. 使用全部 7 个 MOT17 序列 (02/04/05/09/10/11/13)
  2. 仅使用 SDP 检测器变体 (最高质量)
  3. 全局 80/20 train/val split (按 person ID)
  4. 移除离线暗光/遮挡增强 (改为在线数据增强)
  5. 输出: dataset/mot17_reid_all/
============================================================
"""

import os
import sys
import shutil
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image
from tqdm import tqdm


# ============================================================
# [*] 全局路径配置
# ============================================================
BASE_DIR = Path(__file__).parent.resolve()

# MOT17 原始数据集路径
MOT17_RAW_DIR = BASE_DIR / "dataset" / "MOT17" / "MOT17" / "train"

# 输出路径
OUTPUT_DIR = BASE_DIR / "dataset" / "mot17_reid_all"

# ============================================================
# [*] 全部 7 个训练序列 (SDP only)
# ============================================================
#   MOT17-02: 商场室内 (indoor mall)
#   MOT17-04: 步行街夜景 (outdoor night)
#   MOT17-05: 步行街白天 (outdoor day)
#   MOT17-09: 步行街 (outdoor)
#   MOT17-10: 步行街 (outdoor)
#   MOT17-11: 商场室内 (indoor mall)
#   MOT17-13: 车载移动相机 (vehicle-mounted, outdoor)
# ============================================================
SEQUENCE_IDS = ["02", "04", "05", "09", "10", "11", "13"]

# ============================================================
# [*] 清洗过滤参数
# ============================================================
FRAME_INTERVAL = 5         # 5 帧间隔抽帧
MIN_BBOX_SIZE = 30         # 行人框最小尺寸 (像素)
MIN_VISIBILITY = 0.5       # 最低可见度 (遮挡 <= 50%)
ALLOWED_CLASS_IDS = {1}    # 仅保留 class=1 (pedestrian)
TRAIN_VAL_SPLIT = 0.8      # 训练集比例 (train:val = 8:2)
DEDUP_IOU_THRESH = 0.85    # IoU 去重阈值

# ============================================================
# 随机种子
# ============================================================
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ============================================================
# 工具函数
# ============================================================
def compute_iou(box1, box2):
    """计算两个边界框的 IoU"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xi1, yi1 = max(x1, x2), max(y1, y2)
    xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)

    inter_w = max(0, xi2 - xi1)
    inter_h = max(0, yi2 - yi1)
    inter_area = inter_w * inter_h

    area1, area2 = w1 * h1, w2 * h2
    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def parse_gt_file(gt_path):
    """解析 MOT17 gt.txt"""
    frames = defaultdict(list)
    if not gt_path.exists():
        return frames

    with open(gt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 9:
                continue

            frame_id = int(parts[0])
            track_id = int(parts[1])
            x = int(float(parts[2]))
            y = int(float(parts[3]))
            w = int(float(parts[4]))
            h = int(float(parts[5]))
            consider = int(parts[6])
            class_id = int(parts[7])
            visibility = float(parts[8])

            frames[frame_id].append((track_id, x, y, w, h, consider, class_id, visibility))

    return frames


def apply_filter(annotation):
    """过滤检查"""
    _, _, _, w, h, _, class_id, visibility = annotation

    if class_id not in ALLOWED_CLASS_IDS:
        return False
    if visibility < MIN_VISIBILITY:
        return False
    if w < MIN_BBOX_SIZE or h < MIN_BBOX_SIZE:
        return False
    return True


def deduplicate_same_id(entries):
    """同一 ID 连续帧 IoU 去重"""
    if len(entries) <= 1:
        return entries

    deduped = [entries[0]]
    for i in range(1, len(entries)):
        prev_box = deduped[-1][1:]
        curr_box = entries[i][1:]
        iou = compute_iou(prev_box, curr_box)
        if iou < DEDUP_IOU_THRESH:
            deduped.append(entries[i])

    return deduped


def safe_crop_and_save(img, bbox, save_path):
    """安全裁剪并保存行人 ROI"""
    x, y, w, h = bbox
    img_w, img_h = img.size

    x = max(0, x)
    y = max(0, y)
    x2 = min(img_w, x + w)
    y2 = min(img_h, y + h)
    x = min(x, img_w - 1)
    y = min(y, img_h - 1)

    if x2 <= x or y2 <= y:
        return False

    try:
        crop = img.crop((x, y, x2, y2))
        crop.save(save_path, quality=95)
        return True
    except Exception:
        return False


def process_sequence(seq_dir, output_base_dir, seq_cam_id):
    """处理单个 MOT17-SDP 序列"""
    seq_name = seq_dir.name
    img_dir = seq_dir / "img1"
    gt_path = seq_dir / "gt" / "gt.txt"

    if not img_dir.exists():
        print(f"  [ERROR] img1 not found: {img_dir}")
        return None
    if not gt_path.exists():
        print(f"  [ERROR] gt.txt not found: {gt_path}")
        return None

    frames_gt = parse_gt_file(gt_path)
    total_frames = len(frames_gt)
    if total_frames == 0:
        print(f"  [ERROR] Empty gt.txt")
        return None

    print(f"  Total frames: {total_frames}")

    # 按帧间隔采样
    all_frame_ids = sorted(frames_gt.keys())
    sampled_frames = [fid for i, fid in enumerate(all_frame_ids) if i % FRAME_INTERVAL == 0]
    print(f"  Sampled frames: {len(sampled_frames)} (interval={FRAME_INTERVAL})")

    # 按 person ID 收集裁剪信息
    pid_records = defaultdict(list)
    filter_ok = 0
    filter_class = 0
    filter_vis = 0
    filter_size = 0

    for frame_id in sampled_frames:
        annotations = frames_gt.get(frame_id, [])
        for ann in annotations:
            track_id = ann[0]
            if not apply_filter(ann):
                class_id, visibility = ann[6], ann[7]
                if class_id not in ALLOWED_CLASS_IDS:
                    filter_class += 1
                elif visibility < MIN_VISIBILITY:
                    filter_vis += 1
                else:
                    filter_size += 1
                continue
            filter_ok += 1

            # 全局唯一 ID: cam_id * 100000 + track_id
            global_pid = seq_cam_id * 100000 + track_id
            bbox = ann[1:5]
            pid_records[global_pid].append((frame_id,) + bbox)

    print(f"  Filter: ok={filter_ok}, class={filter_class}, vis={filter_vis}, size={filter_size}")

    if not pid_records:
        print(f"  [WARN] No valid pedestrians in {seq_name}")
        return {"records": 0, "person_ids": 0}

    # 去重
    total_before = sum(len(v) for v in pid_records.values())
    for pid in list(pid_records.keys()):
        entries = sorted(pid_records[pid], key=lambda e: e[0])
        pid_records[pid] = deduplicate_same_id(entries)
    total_after = sum(len(v) for v in pid_records.values())
    print(f"  Dedup: {total_before} -> {total_after} (-{total_before - total_after})")

    # 裁剪保存到临时目录
    temp_dir = output_base_dir / "_temp" / seq_name
    temp_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Cropping...")
    crop_ok, crop_fail = 0, 0
    img_cache = {}

    for pid, entries in tqdm(pid_records.items(), desc=f"  {seq_name}", unit="id"):
        pid_dir = temp_dir / str(pid)
        pid_dir.mkdir(exist_ok=True)

        for entry in entries:
            frame_id, x, y, w, h = entry[0], entry[1], entry[2], entry[3], entry[4]

            if frame_id not in img_cache:
                img_path = img_dir / f"{frame_id:06d}.jpg"
                if not img_path.exists():
                    crop_fail += 1
                    continue
                try:
                    img_cache[frame_id] = Image.open(img_path).convert("RGB")
                except Exception:
                    crop_fail += 1
                    continue

            save_name = f"{pid}_c{seq_cam_id}s1_{frame_id:06d}_{crop_ok:04d}.jpg"
            save_path = pid_dir / save_name

            if safe_crop_and_save(img_cache[frame_id], (x, y, w, h), save_path):
                crop_ok += 1
            else:
                crop_fail += 1

    print(f"  Cropped: {crop_ok} ok, {crop_fail} failed")

    return {
        "records": crop_ok,
        "person_ids": len(pid_records),
        "filter_ok": filter_ok,
        "dedup_before": total_before,
        "dedup_after": total_after,
    }


def split_train_val(all_person_dirs, output_dir):
    """按 person ID 全局 80/20 划分"""
    random.shuffle(all_person_dirs)

    n_total = len(all_person_dirs)
    n_train = int(n_total * TRAIN_VAL_SPLIT)

    train_pids = all_person_dirs[:n_train]
    val_pids = all_person_dirs[n_train:]

    print(f"\n[Split] Total IDs: {n_total}, Train: {len(train_pids)}, Val: {len(val_pids)}")

    train_dir = output_dir / "bounding_box_train"
    val_dir = output_dir / "bounding_box_test"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    def copy_images(pid_list, dst_dir, desc):
        count = 0
        for pid_dir, pid in tqdm(pid_list, desc=desc, unit="id"):
            for img_path in pid_dir.glob("*.jpg"):
                try:
                    img = Image.open(img_path).convert("RGB")
                    img.save(dst_dir / img_path.name, quality=95)
                    count += 1
                except Exception:
                    continue
        return count

    print(f"\n[Copy] Train set...")
    train_count = copy_images(train_pids, train_dir, "  Train")
    print(f"  Train images: {train_count}")

    print(f"\n[Copy] Val set...")
    val_count = copy_images(val_pids, val_dir, "  Val")
    print(f"  Val images: {val_count}")

    return train_count, val_count, len(train_pids), len(val_pids)


def write_summary(output_dir, all_stats, train_count, val_count,
                  train_pids_count, val_pids_count):
    """写摘要报告"""
    summary_path = output_dir / "dataset_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("MOT17 All Sequences ReID Dataset Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Output: {output_dir}\n")
        f.write(f"Sequences: {SEQUENCE_IDS} (SDP only)\n")
        f.write(f"Frame interval: {FRAME_INTERVAL}\n")
        f.write(f"Min bbox: {MIN_BBOX_SIZE}x{MIN_BBOX_SIZE}\n")
        f.write(f"Min visibility: {MIN_VISIBILITY}\n")
        f.write(f"Split: {TRAIN_VAL_SPLIT:.0%}/{1 - TRAIN_VAL_SPLIT:.0%}\n\n")

        total_crops = 0
        total_ids = 0
        for seq_name, stats in all_stats.items():
            if stats is None:
                f.write(f"Sequence {seq_name}: FAILED\n\n")
                continue
            f.write(f"Sequence {seq_name}:\n")
            for k, v in stats.items():
                f.write(f"  {k}: {v}\n")
            total_crops += stats.get("records", 0)
            total_ids += stats.get("person_ids", 0)
            f.write("\n")

        f.write("Final Dataset:\n")
        f.write(f"  Total crops: {total_crops}\n")
        f.write(f"  Total person IDs: {total_ids}\n")
        f.write(f"  Train images: {train_count} ({train_pids_count} IDs)\n")
        f.write(f"  Val images:   {val_count} ({val_pids_count} IDs)\n")

    print(f"\n[Summary] Saved: {summary_path}")


def main():
    print("=" * 70)
    print("  MOT17 All Sequences -> Market-1501 ReID Format")
    print(f"  Sequences: {SEQUENCE_IDS} (SDP only)")
    print("=" * 70)

    # ---- 环境检查 ----
    print(f"\n[Path] MOT17 raw: {MOT17_RAW_DIR}")
    print(f"[Path] Output:    {OUTPUT_DIR}")

    if not MOT17_RAW_DIR.exists():
        print(f"\n[ERROR] MOT17 dataset not found: {MOT17_RAW_DIR}")
        sys.exit(1)

    # ---- 查找 SDP 序列 ----
    seq_dirs = []
    for seq_id in SEQUENCE_IDS:
        found = None
        for child in sorted(MOT17_RAW_DIR.iterdir()):
            if child.is_dir() and child.name == f"MOT17-{seq_id}-SDP":
                found = child
                break
        if found:
            seq_dirs.append(found)
        else:
            print(f"  [WARN] MOT17-{seq_id}-SDP not found, skipping")

    if not seq_dirs:
        print(f"\n[ERROR] No SDP sequences found!")
        sys.exit(1)

    print(f"\n[Sequences] Found {len(seq_dirs)} SDP sequences:")
    for d in seq_dirs:
        print(f"  - {d.name}")
    missing = set(SEQUENCE_IDS) - {d.name[5:7] for d in seq_dirs}
    if missing:
        print(f"  Missing: {missing}")

    # ---- 清理旧输出 ----
    if OUTPUT_DIR.exists():
        print(f"\n[Clean] Removing old output: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- 逐序列处理 ----
    all_stats = {}
    all_person_dirs = []

    temp_base = OUTPUT_DIR / "_temp"
    temp_base.mkdir(parents=True, exist_ok=True)

    for cam_idx, seq_dir in enumerate(seq_dirs):
        seq_name = seq_dir.name
        print(f"\n{'─' * 60}")
        print(f"  [{cam_idx + 1}/{len(seq_dirs)}] {seq_name} (cam_id={cam_idx + 1})")
        print(f"{'─' * 60}")

        try:
            stats = process_sequence(seq_dir, OUTPUT_DIR, seq_cam_id=cam_idx + 1)
            all_stats[seq_name] = stats
        except Exception as e:
            print(f"  [EXCEPTION] {e}")
            import traceback
            traceback.print_exc()
            all_stats[seq_name] = None

    # ---- 收集所有行人文件夹 ----
    temp_person_dir = OUTPUT_DIR / "_temp"
    for seq_temp in temp_person_dir.iterdir():
        if seq_temp.is_dir():
            for pid_dir in seq_temp.iterdir():
                if pid_dir.is_dir():
                    pid = int(pid_dir.name)
                    all_person_dirs.append((pid_dir, pid))

    if not all_person_dirs:
        print("\n[ERROR] No valid person data collected!")
        sys.exit(1)

    print(f"\n{'=' * 60}")
    print(f"  Total: {len(all_person_dirs)} person IDs across {len(seq_dirs)} sequences")

    # ---- Train/Val split ----
    train_count, val_count, train_pc, val_pc = split_train_val(all_person_dirs, OUTPUT_DIR)

    # ---- 清理临时目录 ----
    temp_dir = OUTPUT_DIR / "_temp"
    if temp_dir.exists():
        print(f"\n[Clean] Removing temp dir: {temp_dir}")
        shutil.rmtree(temp_dir)

    # ---- 写摘要 ----
    write_summary(OUTPUT_DIR, all_stats, train_count, val_count, train_pc, val_pc)

    # ---- 最终输出 ----
    print(f"\n{'=' * 70}")
    print(f"  [DONE] MOT17 All-Sequence dataset ready!")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Train:  {train_count} images, {train_pc} IDs  -> bounding_box_train/")
    print(f"  Val:    {val_count} images, {val_pc} IDs  -> bounding_box_test/")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
