# -*- coding: utf-8 -*-
"""
酒店监控视频抽帧脚本
功能：从1080P酒店监控视频中每3帧提取1张图片
输入：test_video/hotel_raw/ 下的视频文件
输出：dataset/det/hotel_img/images/
"""

import os
import cv2
import argparse
from pathlib import Path


def extract_frames(video_path, output_dir, frame_interval=3):
    """
    从视频中按间隔抽帧
    :param video_path: 视频文件路径
    :param output_dir: 输出图片目录
    :param frame_interval: 抽帧间隔(每N帧取1帧)
    """
    video_path = Path(video_path)
    if not video_path.exists():
        print(f"[ERROR] 视频文件不存在: {video_path}")
        return 0

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] 无法打开视频: {video_path}")
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[INFO] 视频: {video_path.name}")
    print(f"[INFO] 总帧数: {total_frames}, FPS: {fps:.1f}, 分辨率: {width}x{height}")
    print(f"[INFO] 抽帧间隔: 每{frame_interval}帧取1帧")

    video_name = video_path.stem
    count = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            output_name = f"{video_name}_frame_{count:06d}.jpg"
            output_path = os.path.join(output_dir, output_name)
            cv2.imwrite(output_path, frame)
            count += 1

        frame_idx += 1

    cap.release()
    print(f"[INFO] 抽帧完成，共提取 {count} 张图片\n")
    return count


def main():
    parser = argparse.ArgumentParser(description="酒店监控视频抽帧工具")
    parser.add_argument("--input_dir", type=str,
                        default="Hotel_Exp/test_video/hotel_raw",
                        help="输入视频目录")
    parser.add_argument("--output_dir", type=str,
                        default="Hotel_Exp/dataset/det/hotel_img/images",
                        help="输出图片目录")
    parser.add_argument("--interval", type=int, default=3,
                        help="抽帧间隔帧数")
    parser.add_argument("--ext", type=str, default=".mp4,.avi,.mov,.mkv",
                        help="视频文件扩展名")

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        print(f"[ERROR] 输入目录不存在: {input_dir}")
        print("[INFO] 请将酒店监控视频放入该目录后重新运行")
        return

    exts = args.ext.split(",")
    video_files = []
    for ext in exts:
        video_files.extend(input_dir.glob(f"*{ext.strip()}"))

    if not video_files:
        print(f"[WARN] 输入目录中未找到视频文件，支持的格式: {exts}")
        return

    print(f"[INFO] 找到 {len(video_files)} 个视频文件，开始抽帧...\n")
    total_images = 0

    for video_path in sorted(video_files):
        count = extract_frames(video_path, str(output_dir), args.interval)
        total_images += count

    print(f"[INFO] 全部完成，共抽取 {total_images} 张图片")


if __name__ == "__main__":
    main()