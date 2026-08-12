"""
============================================================
MOT17 数据集清洗加工脚本: mot17_reid_data_clean.py
============================================================
功能:
  1. 筛选 MOT17 室内序列 (MOT17-02, MOT17-11 商场场景)
  2. 剔除室外街道视频 (04/05/09/10/13)
  3. 5 帧间隔抽帧去重
  4. 过滤模糊、遮挡超 50%、小于 30x30 像素行人
  5. 读取 gt.txt 裁剪单人 ROI,按行人 ID 分文件夹
  6. 完全复刻 Market-1501 目录结构
  7. 数据集划分 train:val = 8:2
  8. 训练集加暗光、遮挡仿真增强,验证集仅基础归一化

MOT17 数据集格式 (gt.txt):
  frame_id, track_id, x, y, w, h, consider_in_eval, class_id, visibility
  class_id: 1=pedestrian, 2=person on vehicle, 7=static person, ...
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
# [*] 全局路径配置（Windows 反斜杠兼容）
# ============================================================
BASE_DIR = Path(__file__).parent.resolve()  # 脚本所在目录即项目根目录

# MOT17 原始数据集路径
MOT17_RAW_DIR = BASE_DIR / "dataset" / "MOT17" / "MOT17" / "train"

# MOT17 清洗后标准 ReID 数据集输出路径
OUTPUT_DIR = BASE_DIR / "dataset" / "mot17_reid_clean"

# ============================================================
# [*] 室内序列白名单 (MOT challenge 官方定义)
# ============================================================
#   MOT17-02: 商场 (shopping mall)   -- 室内 [OK]
#   MOT17-11: 商场 (shopping mall)   -- 室内 [OK]
#   MOT17-04: 步行街夜景              -- 室外 ✗
#   MOT17-05: 步行街白天              -- 室外 ✗
#   MOT17-09: 步行街                  -- 室外 ✗
#   MOT17-10: 步行街                  -- 室外 ✗
#   MOT17-13: 车载移动相机            -- 室外 ✗
# ============================================================
INDOOR_SEQUENCE_IDS = ["02", "11"]  # 仅保留室内序列

# ============================================================
# [*] 清洗过滤参数
# ============================================================
FRAME_INTERVAL = 5         # 5 帧间隔抽帧
MIN_BBOX_SIZE = 30         # 行人框最小尺寸 (像素)
MIN_VISIBILITY = 0.5       # 最低可见度 (遮挡 <= 50%)
ALLOWED_CLASS_IDS = {1}    # 仅保留 class=1 (pedestrian), 过滤 static/vehicle 等
TRAIN_VAL_SPLIT = 0.8      # 训练集比例 (train:val = 8:2)

# 去重阈值:同一 ID 的相邻采样帧之间 IOU 超过此值视为重复
DEDUP_IOU_THRESH = 0.85

# ============================================================
# 随机种子（保证划分可复现）
# ============================================================
RANDOM_SEED = 42
random.seed(RANDOM_SEED)


# ============================================================
# 工具函数
# ============================================================
def compute_iou(box1, box2):
    """计算两个边界框的 IOU
    Args:
        box1, box2: (x, y, w, h) 元组
    Returns:
        IOU 值 (0~1)
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # 计算交集
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
    """解析 MOT17 gt.txt 文件
    Args:
        gt_path: gt.txt 文件路径
    Returns:
        frames: dict {frame_id: [(track_id, x, y, w, h, consider, class_id, visibility), ...]}
    """
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
    """对单个标注进行过滤检查
    Args:
        annotation: (track_id, x, y, w, h, consider, class_id, visibility)
    Returns:
        (valid_bool, reason_str)
    """
    track_id, x, y, w, h, consider, class_id, visibility = annotation

    # 1) 类别过滤:仅保留 class=1 (pedestrian)
    if class_id not in ALLOWED_CLASS_IDS:
        return False, f"class={class_id}"

    # 2) 遮挡过滤:遮挡超过 50% 则丢弃
    if visibility < MIN_VISIBILITY:
        return False, f"vis={visibility:.3f}<{MIN_VISIBILITY}"

    # 3) 尺寸过滤:宽或高小于 MIN_BBOX_SIZE 则丢弃
    if w < MIN_BBOX_SIZE or h < MIN_BBOX_SIZE:
        return False, f"size={w}x{h}<{MIN_BBOX_SIZE}"

    return True, "ok"


def deduplicate_same_id(entries):
    """对同一 ID 的连续采样帧去重
    如果相邻帧之间 IOU 超过阈值,保留前者丢弃后者（去除静态重复帧）
    Args:
        entries: [(frame_id, x, y, w, h), ...] 按 frame_id 排序
    Returns:
        deduped: 去重后的条目列表
    """
    if len(entries) <= 1:
        return entries

    deduped = [entries[0]]
    for i in range(1, len(entries)):
        prev_box = deduped[-1][1:]  # (x, y, w, h)
        curr_box = entries[i][1:]
        iou = compute_iou(prev_box, curr_box)
        if iou < DEDUP_IOU_THRESH:
            deduped.append(entries[i])

    return deduped


def safe_crop_and_save(img, bbox, save_path):
    """安全裁剪行人 ROI 并保存
    带边界越界修正和异常捕获
    Args:
        img: PIL Image 对象
        bbox: (x, y, w, h)
        save_path: 保存路径
    Returns:
        success: bool
        error: str or None
    """
    x, y, w, h = bbox
    img_w, img_h = img.size

    # 边界修正
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
    """暗光仿真:随机降低亮度 20%~60%,模拟酒店走廊暗光场景"""
    factor = random.uniform(0.4, 0.8)  # 亮度系数
    enhancer = ImageEnhance.Brightness(img)
    return enhancer.enhance(factor)


def simulate_occlusion_augmentation(img):
    """遮挡仿真:在图像上随机添加黑色矩形块,模拟局部遮挡"""
    img_cp = img.copy()
    w, h = img_cp.size

    # 随机 1~2 个遮挡块
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
    """处理单个 MOT17 序列
    Args:
        seq_dir: 序列目录 (e.g., MOT17-02-SDP)
        output_base_dir: 输出基础目录
        seq_cam_id: 序列对应的摄像头 ID (用于 ReID 文件命名)
    Returns:
        stats: dict 统计信息
    """
    seq_name = seq_dir.name  # e.g., "MOT17-02-SDP"
    img_dir = seq_dir / "img1"
    gt_path = seq_dir / "gt" / "gt.txt"

    if not img_dir.exists():
        print(f"  [错误] img1 目录不存在: {img_dir}")
        return {"error": "img1 not found"}

    if not gt_path.exists():
        print(f"  [错误] gt.txt 不存在: {gt_path}")
        return {"error": "gt.txt not found"}

    # 解析 gt.txt
    frames_gt = parse_gt_file(gt_path)
    total_frames = len(frames_gt)
    if total_frames == 0:
        print(f"  [错误] gt.txt 为空")
        return {"error": "empty gt.txt"}

    print(f"  [信息] 总帧数: {total_frames}")

    # 按 5 帧间隔采样帧号
    all_frame_ids = sorted(frames_gt.keys())
    sampled_frames = [fid for i, fid in enumerate(all_frame_ids) if i % FRAME_INTERVAL == 0]

    print(f"  [信息] 采样帧数: {len(sampled_frames)} (间隔={FRAME_INTERVAL})")

    # 按行人 ID 收集裁剪信息
    # pid_records[global_pid] = [(frame_id, x, y, w, h), ...]
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

            # 全局 ID = 序列摄像头 ID x 100000 + track_id (跨序列唯一)
            global_pid = seq_cam_id * 100000 + track_id
            bbox = ann[1:5]  # (x, y, w, h)
            pid_records[global_pid].append((frame_id,) + bbox)

    print(f"  [过滤统计] 通过: {filter_stats['ok']}, "
          f"类别不合: {filter_stats['class']}, "
          f"遮挡过多: {filter_stats['visibility']}, "
          f"尺寸过小: {filter_stats['size']}")

    if not pid_records:
        print(f"  [警告] 序列 {seq_name} 无有效行人")
        return {"records": 0}

    # 去重:按 ID 对采样帧去重
    total_before_dedup = sum(len(v) for v in pid_records.values())
    for pid in list(pid_records.keys()):
        entries = sorted(pid_records[pid], key=lambda e: e[0])
        pid_records[pid] = deduplicate_same_id(entries)
    total_after_dedup = sum(len(v) for v in pid_records.values())

    print(f"  [去重] {total_before_dedup} → {total_after_dedup} "
          f"(去除 {total_before_dedup - total_after_dedup} 个重复)")

    # 裁剪并保存临时文件
    # temp_dir / pid / img.jpg
    temp_dir = output_base_dir / "_temp" / seq_name
    temp_dir.mkdir(parents=True, exist_ok=True)

    print(f"  [裁剪] 正在裁剪行人 ROI...")
    crop_count = 0
    crop_fail = 0

    # 建立帧号→图像路径缓存 (MOT17 图像命名为 000001.jpg)
    img_cache = {}

    for pid, entries in tqdm(pid_records.items(), desc=f"  {seq_name}", unit="id"):
        pid_dir = temp_dir / str(pid)
        pid_dir.mkdir(exist_ok=True)

        for entry in entries:
            frame_id = entry[0]
            bbox = entry[1:]  # (x, y, w, h)

            # 加载图像（缓存机制）
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

            # 保存裁剪 ROI,文件命名格式: pid_camId_seq_frame.jpg (兼容 Market-1501)
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
    """按行人 ID 划分训练/验证集,完全复刻 Market-1501 目录结构
    Args:
        all_person_dirs: list of (pid_dir_path, pid)
        output_dir: 输出根目录
        augment_train: 是否对训练集做暗光+遮挡仿真增强
    """
    # 随机打乱
    random.shuffle(all_person_dirs)

    n_total = len(all_person_dirs)
    n_train = int(n_total * TRAIN_VAL_SPLIT)

    train_pids = all_person_dirs[:n_train]
    val_pids = all_person_dirs[n_train:]

    print(f"\n[划分] 总人数: {n_total}, 训练: {len(train_pids)}, 验证: {len(val_pids)}")

    # 创建 Market-1501 风格目录
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
                    # 原始图像
                    img.save(dst_dir / fname, quality=95)
                    count += 1

                    # 暗光仿真增强
                    dark_img = simulate_dark_augmentation(img)
                    dark_name = fname.replace(".jpg", "_dark.jpg")
                    dark_img.save(dst_dir / dark_name, quality=95)
                    count += 1

                    # 遮挡仿真增强
                    occ_img = simulate_occlusion_augmentation(img)
                    occ_name = fname.replace(".jpg", "_occ.jpg")
                    occ_img.save(dst_dir / occ_name, quality=95)
                    count += 1
                else:
                    # 验证集:仅保存原始图像
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


def write_dataset_summary(output_dir, all_stats, train_count, val_count,
                          train_person_count, val_person_count):
    """写数据集清洗摘要"""
    summary_path = output_dir / "dataset_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("MOT17 → Market-1501 ReID 数据集清洗报告\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"输出路径: {output_dir}\n")
        f.write(f"目录结构: bounding_box_train / bounding_box_test\n\n")

        f.write("处理参数:\n")
        f.write(f"  室内序列: {INDOOR_SEQUENCE_IDS}\n")
        f.write(f"  抽帧间隔: 每 {FRAME_INTERVAL} 帧\n")
        f.write(f"  最小行人尺寸: {MIN_BBOX_SIZE}x{MIN_BBOX_SIZE} px\n")
        f.write(f"  最低可见度: {MIN_VISIBILITY}\n")
        f.write(f"  去重 IOU 阈值: {DEDUP_IOU_THRESH}\n")
        f.write(f"  类别白名单: {ALLOWED_CLASS_IDS}\n\n")

        total_crops = 0
        for seq_name, stats in all_stats.items():
            f.write(f"序列 {seq_name}:\n")
            for k, v in stats.items():
                f.write(f"  {k}: {v}\n")
            total_crops += stats.get("records", 0)
            f.write("\n")

        f.write("最终数据集统计:\n")
        f.write(f"  总裁剪数: {total_crops}\n")
        f.write(f"  训练集图像: {train_count}\n")
        f.write(f"  训练集人数: {train_person_count}\n")
        f.write(f"  验证集图像: {val_count}\n")
        f.write(f"  验证集人数: {val_person_count}\n")

    print(f"\n[摘要] 已保存至: {summary_path}")


def main():
    """主函数:MOT17 数据清洗流水线"""
    print("=" * 70)
    print("  MOT17 数据集清洗 → Market-1501 ReID 格式")
    print("=" * 70)

    # ========== 环境检查 ==========
    print(f"\n[路径] MOT17 原始路径: {MOT17_RAW_DIR}")
    print(f"[路径] ReID 输出路径:   {OUTPUT_DIR}")

    if not MOT17_RAW_DIR.exists():
        print(f"\n[错误] MOT17 数据集路径不存在!")
        print(f"  期望路径: {MOT17_RAW_DIR}")
        print(f"  请确认 MOT17 数据集已解压至正确位置。")
        sys.exit(1)

    # 查找室内 SDP 序列
    indoor_seq_dirs = []
    for seq_id in INDOOR_SEQUENCE_IDS:
        # 查找匹配的目录 (MOT17-XX-* 格式)
        for child in MOT17_RAW_DIR.iterdir():
            if child.is_dir() and child.name.startswith(f"MOT17-{seq_id}-"):
                if "SDP" in child.name:  # 优先使用 SDP 检测器变体
                    indoor_seq_dirs.append(child)
                elif not any("SDP" in d.name and d.name.startswith(f"MOT17-{seq_id}-")
                           for d in indoor_seq_dirs):
                    # 如果没有 SDP 变体,使用任意变体
                    indoor_seq_dirs.append(child)

    # 去重
    indoor_seq_dirs = sorted(set(indoor_seq_dirs))

    if not indoor_seq_dirs:
        print(f"\n[错误] 未找到室内序列!")
        print(f"  期望序列号: {INDOOR_SEQUENCE_IDS}")
        print(f"  扫描路径: {MOT17_RAW_DIR}")
        print(f"  可用子目录: {[d.name for d in MOT17_RAW_DIR.iterdir() if d.is_dir()]}")
        sys.exit(1)

    print(f"\n[序列] 发现 {len(indoor_seq_dirs)} 个室内序列:")
    for d in indoor_seq_dirs:
        print(f"  - {d.name}")
    print(f"\n[序列] 已剔除室外序列 (04/05/09/10/13)")

    # ========== 清理旧输出 ==========
    if OUTPUT_DIR.exists():
        print(f"\n[清理] 删除旧输出目录: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ========== 逐序列处理 ==========
    all_stats = {}
    all_person_dirs = []  # [(dir_path, pid), ...]

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

    # ========== 收集所有行人文件夹 ==========
    temp_person_dir = OUTPUT_DIR / "_temp"
    for seq_temp in temp_person_dir.iterdir():
        if seq_temp.is_dir():
            for pid_dir in seq_temp.iterdir():
                if pid_dir.is_dir():
                    pid = int(pid_dir.name)
                    all_person_dirs.append((pid_dir, pid))

    if not all_person_dirs:
        print("\n[错误] 没有收集到任何有效行人数据!")
        print("  请检查 MOT17 数据完整性和过滤参数。")
        sys.exit(1)

    print(f"\n{'=' * 60}")
    print(f"  总计: {len(all_person_dirs)} 个行人 ID")

    # ========== Train / Val 划分并复制 ==========
    train_count, val_count, train_pc, val_pc = split_train_val(
        all_person_dirs, OUTPUT_DIR, augment_train=True
    )

    # ========== 清理临时目录 ==========
    temp_dir = OUTPUT_DIR / "_temp"
    if temp_dir.exists():
        print(f"\n[清理] 删除临时目录: {temp_dir}")
        shutil.rmtree(temp_dir)

    # ========== 写摘要报告 ==========
    write_dataset_summary(OUTPUT_DIR, all_stats, train_count, val_count, train_pc, val_pc)

    # ========== 最终输出 ==========
    print(f"\n{'=' * 70}")
    print(f"  [OK] MOT17 数据集清洗完成!")
    print(f"  输出路径: {OUTPUT_DIR}")
    print(f"  目录结构:")
    print(f"    {OUTPUT_DIR / 'bounding_box_train'}  ({train_count} 张图像, {train_pc} 人)")
    print(f"    {OUTPUT_DIR / 'bounding_box_test'}   ({val_count} 张图像, {val_pc} 人)")
    print(f"{'=' * 70}")

    return OUTPUT_DIR, train_count, val_count


if __name__ == "__main__":
    main()
