"""
============================================================
Kaggle 版: MOT17 数据集清洗加工脚本
适配 Kaggle 输入/输出路径
============================================================
用法: 在 Kaggle Notebook 中运行此脚本
输出: /kaggle/working/mot17_reid_clean/ (Market-1501 格式)
============================================================
"""

import os
import sys
import shutil
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image, ImageEnhance
from tqdm import tqdm


# ============================================================
# [*] Kaggle 路径配置
# ============================================================
# MOT17 原始数据集 (Kaggle 输入，只读)
# 数据集解压后可能是: mot17-zip/MOT17/train/ 或 mot17-zip/train/
MOT17_DATASET_DIR = Path("/kaggle/input/datasets/ahmedalycess/mot17-zip")

# MOT17 清洗后标准 ReID 数据集输出路径 (Kaggle 工作目录，可写)
OUTPUT_DIR = Path("/kaggle/working/mot17_reid_clean")

# ── 自动探测 MOT17 train 目录 ──
def find_mot17_train_dir(base_dir):
    """从数据集根目录自动定位 MOT17 的 train 目录"""
    candidates = [
        base_dir / "MOT17" / "train",      # mot17-zip/MOT17/train
        base_dir / "train",                # mot17-zip/train
        base_dir / "MOT17",                # mot17-zip/MOT17
        base_dir,                          # 根目录本身
    ]
    for cand in candidates:
        if cand.exists() and cand.is_dir():
            for child in cand.iterdir():
                if child.is_dir() and child.name.startswith("MOT17-"):
                    print(f"  [探测] 找到 MOT17 train 目录: {cand}")
                    return cand
    return None

MOT17_RAW_DIR = find_mot17_train_dir(MOT17_DATASET_DIR)

# ============================================================
# [*] 室内序列白名单
# ============================================================
INDOOR_SEQUENCE_IDS = ["02", "11"]

# ============================================================
# [*] 清洗过滤参数
# ============================================================
FRAME_INTERVAL = 5
MIN_BBOX_SIZE = 30
MIN_VISIBILITY = 0.5
ALLOWED_CLASS_IDS = {1}
TRAIN_VAL_SPLIT = 0.8
DEDUP_IOU_THRESH = 0.85
RANDOM_SEED = 42
random.seed(RANDOM_SEED)


# ============================================================
# 工具函数 (与原版相同)
# ============================================================
def compute_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    inter_w = max(0, xi2 - xi1)
    inter_h = max(0, yi2 - yi1)
    inter_area = inter_w * inter_h
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def parse_gt_file(gt_path):
    frames = defaultdict(list)
    if not gt_path.exists():
        print(f"  [警告] gt.txt 不存在: {gt_path}", flush=True)
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


def apply_filter(annotation, frame_id):
    track_id, x, y, w, h, consider, class_id, visibility = annotation
    if class_id not in ALLOWED_CLASS_IDS:
        return False, f"class={class_id}"
    if visibility < MIN_VISIBILITY:
        return False, f"vis={visibility:.3f}<{MIN_VISIBILITY}"
    if w < MIN_BBOX_SIZE or h < MIN_BBOX_SIZE:
        return False, f"size={w}x{h}<{MIN_BBOX_SIZE}"
    return True, "ok"


def deduplicate_same_id(entries):
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
    x, y, w, h = bbox
    img_w, img_h = img.size
    x = max(0, x)
    y = max(0, y)
    x2 = min(img_w, x + w)
    y2 = min(img_h, y + h)
    x = min(x, img_w - 1)
    y = min(y, img_h - 1)
    if x2 <= x or y2 <= y:
        return False, f"invalid crop: ({x},{y})-({x2},{y2})"
    try:
        crop = img.crop((x, y, x2, y2))
        crop.save(save_path, quality=95)
        return True, None
    except Exception as e:
        return False, str(e)


def simulate_dark_augmentation(img):
    factor = random.uniform(0.4, 0.8)
    enhancer = ImageEnhance.Brightness(img)
    return enhancer.enhance(factor)


def simulate_occlusion_augmentation(img):
    img_cp = img.copy()
    w, h = img_cp.size
    num_blocks = random.randint(1, 2)
    for _ in range(num_blocks):
        block_w = random.randint(int(w * 0.1), int(w * 0.35))
        block_h = random.randint(int(h * 0.1), int(h * 0.35))
        block_x = random.randint(0, max(1, w - block_w))
        block_y = random.randint(0, max(1, h - block_h))
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img_cp)
        draw.rectangle([block_x, block_y, block_x + block_w, block_y + block_h],
                       fill=(0, 0, 0))
    return img_cp


def process_sequence(seq_dir, output_base_dir, seq_cam_id):
    seq_name = seq_dir.name
    img_dir = seq_dir / "img1"
    gt_path = seq_dir / "gt" / "gt.txt"

    if not img_dir.exists():
        print(f"  [错误] img1 目录不存在: {img_dir}")
        return {"error": "img1 not found"}
    if not gt_path.exists():
        print(f"  [错误] gt.txt 不存在: {gt_path}")
        return {"error": "gt.txt not found"}

    frames_gt = parse_gt_file(gt_path)
    total_frames = len(frames_gt)
    if total_frames == 0:
        print(f"  [错误] gt.txt 为空")
        return {"error": "empty gt.txt"}
    print(f"  [信息] 总帧数: {total_frames}")

    all_frame_ids = sorted(frames_gt.keys())
    sampled_frames = [fid for i, fid in enumerate(all_frame_ids) if i % FRAME_INTERVAL == 0]
    print(f"  [信息] 采样帧数: {len(sampled_frames)} (间隔={FRAME_INTERVAL})")

    pid_records = defaultdict(list)
    filter_stats = {"class": 0, "visibility": 0, "size": 0, "ok": 0}

    for frame_id in sampled_frames:
        annotations = frames_gt.get(frame_id, [])
        for ann in annotations:
            track_id = ann[0]
            valid, reason = apply_filter(ann, frame_id)
            if not valid:
                if "class" in reason:
                    filter_stats["class"] += 1
                elif "vis" in reason:
                    filter_stats["visibility"] += 1
                elif "size" in reason:
                    filter_stats["size"] += 1
                continue
            filter_stats["ok"] += 1
            global_pid = seq_cam_id * 100000 + track_id
            bbox = ann[1:5]
            pid_records[global_pid].append((frame_id,) + bbox)

    print(f"  [过滤统计] 通过: {filter_stats['ok']}, "
          f"类别不合: {filter_stats['class']}, "
          f"遮挡过多: {filter_stats['visibility']}, "
          f"尺寸过小: {filter_stats['size']}")

    if not pid_records:
        print(f"  [警告] 序列 {seq_name} 无有效行人")
        return {"records": 0}

    total_before_dedup = sum(len(v) for v in pid_records.values())
    for pid in list(pid_records.keys()):
        entries = sorted(pid_records[pid], key=lambda e: e[0])
        pid_records[pid] = deduplicate_same_id(entries)
    total_after_dedup = sum(len(v) for v in pid_records.values())
    print(f"  [去重] {total_before_dedup} → {total_after_dedup} "
          f"(去除 {total_before_dedup - total_after_dedup} 个重复)")

    temp_dir = output_base_dir / "_temp" / seq_name
    temp_dir.mkdir(parents=True, exist_ok=True)

    print(f"  [裁剪] 正在裁剪行人 ROI...")
    crop_count = 0
    crop_fail = 0
    img_cache = {}

    for pid, entries in tqdm(pid_records.items(), desc=f"  {seq_name}", unit="id"):
        pid_dir = temp_dir / str(pid)
        pid_dir.mkdir(exist_ok=True)
        for entry in entries:
            frame_id = entry[0]
            bbox = entry[1:]
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
            img = img_cache[frame_id]
            save_name = f"{pid}_c{seq_cam_id}s1_{frame_id:06d}_{crop_count:02d}.jpg"
            save_path = pid_dir / save_name
            success, err = safe_crop_and_save(img, bbox, save_path)
            if success:
                crop_count += 1
            else:
                crop_fail += 1

    print(f"  [裁剪] 成功: {crop_count}, 失败: {crop_fail}")
    return {
        "records": crop_count,
        "person_ids": len(pid_records),
        "filtered_ok": filter_stats["ok"],
        "dedup_before": total_before_dedup,
        "dedup_after": total_after_dedup,
    }


def split_train_val(all_person_dirs, output_dir, augment_train=True):
    random.shuffle(all_person_dirs)
    n_total = len(all_person_dirs)
    n_train = int(n_total * TRAIN_VAL_SPLIT)
    train_pids = all_person_dirs[:n_train]
    val_pids = all_person_dirs[n_train:]
    print(f"\n[划分] 总人数: {n_total}, 训练: {len(train_pids)}, 验证: {len(val_pids)}")

    train_dir = output_dir / "bounding_box_train"
    val_dir = output_dir / "bounding_box_test"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    def copy_person_images(pid_list, dst_dir, do_augment):
        count = 0
        for pid_dir, pid in tqdm(pid_list, desc=f"  {'增强' if do_augment else '复刻'}", unit="id"):
            img_files = list(pid_dir.glob("*.jpg"))
            for img_path in img_files:
                try:
                    img = Image.open(img_path).convert("RGB")
                except Exception:
                    continue
                fname = img_path.name
                if do_augment:
                    img.save(dst_dir / fname, quality=95)
                    count += 1
                    dark_img = simulate_dark_augmentation(img)
                    dark_name = fname.replace(".jpg", "_dark.jpg")
                    dark_img.save(dst_dir / dark_name, quality=95)
                    count += 1
                    occ_img = simulate_occlusion_augmentation(img)
                    occ_name = fname.replace(".jpg", "_occ.jpg")
                    occ_img.save(dst_dir / occ_name, quality=95)
                    count += 1
                else:
                    img.save(dst_dir / fname, quality=95)
                    count += 1
        return count

    print(f"\n[复制] 训练集（含暗光+遮挡仿真增强）...")
    train_count = copy_person_images(train_pids, train_dir, do_augment=True)
    print(f"  训练集图像数: {train_count}")
    print(f"\n[复制] 验证集（仅基础复刻）...")
    val_count = copy_person_images(val_pids, val_dir, do_augment=False)
    print(f"  验证集图像数: {val_count}")
    return train_count, val_count, len(train_pids), len(val_pids)


def main():
    print("=" * 70)
    print("  Kaggle: MOT17 数据集清洗 → Market-1501 ReID 格式")
    print("=" * 70)

    print(f"\n[路径] MOT17 数据集:    {MOT17_DATASET_DIR}")
    print(f"[路径] MOT17 train 目录: {MOT17_RAW_DIR}")
    print(f"[路径] ReID 输出路径:   {OUTPUT_DIR}")

    if MOT17_RAW_DIR is None or not MOT17_RAW_DIR.exists():
        print(f"\n[错误] 无法定位 MOT17 train 目录!")
        print(f"  在 {MOT17_DATASET_DIR} 下未找到包含 MOT17-XX-* 序列的 train 目录。")
        print(f"  请检查数据集目录结构。")
        kaggle_input = Path("/kaggle/input")
        for d in kaggle_input.iterdir():
            if d.is_dir():
                print(f"  /kaggle/input 子目录: {d}")
                for sub in list(d.iterdir())[:10]:
                    print(f"    - {sub.name}")
        sys.exit(1)

    # 查找室内 SDP 序列
    indoor_seq_dirs = []
    for seq_id in INDOOR_SEQUENCE_IDS:
        for child in MOT17_RAW_DIR.iterdir():
            if child.is_dir() and child.name.startswith(f"MOT17-{seq_id}-"):
                if "SDP" in child.name:
                    indoor_seq_dirs.append(child)
                elif not any("SDP" in d.name and d.name.startswith(f"MOT17-{seq_id}-")
                           for d in indoor_seq_dirs):
                    indoor_seq_dirs.append(child)
    indoor_seq_dirs = sorted(set(indoor_seq_dirs))

    if not indoor_seq_dirs:
        print(f"\n[错误] 未找到室内序列!")
        print(f"  期望序列号: {INDOOR_SEQUENCE_IDS}")
        print(f"  扫描路径: {MOT17_RAW_DIR}")
        available = [d.name for d in MOT17_RAW_DIR.iterdir() if d.is_dir()]
        print(f"  可用子目录: {available}")
        # 尝试自动检测：可能没有 SDP 后缀
        for seq_id in INDOOR_SEQUENCE_IDS:
            for child in MOT17_RAW_DIR.iterdir():
                if child.is_dir() and seq_id in child.name:
                    indoor_seq_dirs.append(child)
        indoor_seq_dirs = sorted(set(indoor_seq_dirs))

    if not indoor_seq_dirs:
        sys.exit(1)

    print(f"\n[序列] 发现 {len(indoor_seq_dirs)} 个室内序列:")
    for d in indoor_seq_dirs:
        print(f"  - {d.name}")

    # 清理旧输出
    if OUTPUT_DIR.exists():
        print(f"\n[清理] 删除旧输出目录: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 逐序列处理
    all_stats = {}
    all_person_dirs = []
    temp_base = OUTPUT_DIR / "_temp"
    temp_base.mkdir(parents=True, exist_ok=True)

    for cam_idx, seq_dir in enumerate(indoor_seq_dirs):
        seq_name = seq_dir.name
        print(f"\n{'-' * 60}")
        print(f"  处理序列: {seq_name} (cam_id={cam_idx + 1})")
        print(f"{'-' * 60}")
        try:
            stats = process_sequence(seq_dir, OUTPUT_DIR, seq_cam_id=cam_idx + 1)
            all_stats[seq_name] = stats
        except Exception as e:
            print(f"  [异常] 处理序列 {seq_name} 时出错: {e}")
            import traceback
            traceback.print_exc()
            all_stats[seq_name] = {"error": str(e)}

    # 收集所有行人文件夹
    temp_person_dir = OUTPUT_DIR / "_temp"
    for seq_temp in temp_person_dir.iterdir():
        if seq_temp.is_dir():
            for pid_dir in seq_temp.iterdir():
                if pid_dir.is_dir():
                    pid = int(pid_dir.name)
                    all_person_dirs.append((pid_dir, pid))

    if not all_person_dirs:
        print("\n[错误] 没有收集到任何有效行人数据!")
        sys.exit(1)

    print(f"\n{'=' * 60}")
    print(f"  总计: {len(all_person_dirs)} 个行人 ID")

    # Train/Val 划分
    train_count, val_count, train_pc, val_pc = split_train_val(
        all_person_dirs, OUTPUT_DIR, augment_train=True
    )

    # 清理临时目录
    temp_dir = OUTPUT_DIR / "_temp"
    if temp_dir.exists():
        print(f"\n[清理] 删除临时目录: {temp_dir}")
        shutil.rmtree(temp_dir)

    print(f"\n{'=' * 70}")
    print(f"  [OK] MOT17 数据集清洗完成!")
    print(f"  输出路径: {OUTPUT_DIR}")
    print(f"  bounding_box_train: {train_count} 张, {train_pc} 人")
    print(f"  bounding_box_test:  {val_count} 张, {val_pc} 人")
    print(f"{'=' * 70}")

    return OUTPUT_DIR, train_count, val_count


if __name__ == "__main__":
    main()
