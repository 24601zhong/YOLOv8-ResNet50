# -*- coding: utf-8 -*-
"""
酒店异常人员实时监控识别系统 - 主运行脚本 run_all.py
======================================================
执行流程（按顺序）：
  Phase 0: 环境校验与初始化
  Phase 1: 数据集制备(抽帧->清洗->标注->划分->增强)
  Phase 2: YOLOv8改进模型训练与消融实验
  Phase 3: ResNet50重识别模型训练与阈值实验
  Phase 4: Flask系统启动
  Phase 5: 全链路集成测试

用法:
  python run_all.py --phase all          # 执行全部阶段
  python run_all.py --phase dataset      # 仅执行数据集制备
  python run_all.py --phase yolo         # 仅执行YOLOv8训练
  python run_all.py --phase reid         # 仅执行ResNet50训练
  python run_all.py --phase system       # 仅启动Flask系统
  python run_all.py --phase env_check    # 仅环境校验
"""

import os
import sys
import time
import argparse
import subprocess
import json
from pathlib import Path
from datetime import datetime

# 项目根目录
PROJECT_ROOT = Path(__file__).parent
os.chdir(PROJECT_ROOT)

# 日志颜色
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    RESET = '\033[0m'


def log_info(msg):
    print(f"{Colors.BLUE}[INFO]{Colors.RESET} {msg}")

def log_success(msg):
    print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} {msg}")

def log_warning(msg):
    print(f"{Colors.YELLOW}[WARNING]{Colors.RESET} {msg}")

def log_error(msg):
    print(f"{Colors.RED}[ERROR]{Colors.RESET} {msg}")

def log_step(phase, msg):
    print(f"\n{Colors.CYAN}{'='*60}")
    print(f"  Phase {phase}: {msg}")
    print(f"{'='*60}{Colors.RESET}\n")


def run_command(cmd, cwd=None, timeout=300):
    """执行命令并返回结果"""
    log_info(f"执行: {cmd}")
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            cwd=cwd or str(PROJECT_ROOT),
            timeout=timeout
        )
        if result.stdout:
            print(result.stdout.strip())
        if result.stderr and result.returncode != 0:
            log_warning(result.stderr.strip())
        return result.returncode == 0, result
    except subprocess.TimeoutExpired:
        log_error(f"命令超时: {cmd}")
        return False, None
    except Exception as e:
        log_error(f"命令执行失败: {str(e)}")
        return False, None


# ============================================================
# Phase 0: 环境校验
# ============================================================

def phase0_env_check():
    """环境校验"""
    log_step(0, "环境校验与初始化")

    # 检查Python版本
    python_ver = sys.version_info
    log_info(f"Python版本: {python_ver.major}.{python_ver.minor}.{python_ver.micro}")

    # 检查关键库
    required_libs = ['torch', 'cv2', 'numpy', 'flask', 'ultralytics', 'albumentations']
    missing = []

    for lib in required_libs:
        try:
            if lib == 'cv2':
                import cv2
                log_success(f"  OpenCV: {cv2.__version__}")
            elif lib == 'torch':
                import torch
                log_success(f"  PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
                if torch.cuda.is_available():
                    log_info(f"  GPU: {torch.cuda.get_device_name(0)}")
                    log_info(f"  显存: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")
            elif lib == 'flask':
                import flask
                log_success(f"  Flask: {flask.__version__}")
            elif lib == 'ultralytics':
                import ultralytics
                log_success(f"  Ultralytics: {ultralytics.__version__}")
            elif lib == 'albumentations':
                import albumentations
                log_success(f"  Albumentations: {albumentations.__version__}")
            else:
                lib_mod = __import__(lib)
                log_success(f"  {lib}: OK")
        except ImportError as e:
            missing.append(lib)
            log_warning(f"  缺少库: {lib} - {str(e)}")

    if missing:
        log_warning(f"缺少 {len(missing)} 个库，请运行: pip install -r requirements.txt")
    else:
        log_success("所有关键库已安装")

    # 检查目录结构
    required_dirs = [
        'dataset/det', 'dataset/reid',
        'yolov8_exp', 'resnet50_reid',
        'system_server', 'test_video',
        'output'
    ]

    for d in required_dirs:
        dir_path = PROJECT_ROOT / d
        if dir_path.exists():
            log_success(f"  目录存在: {d}")
        else:
            dir_path.mkdir(parents=True, exist_ok=True)
            log_info(f"  创建目录: {d}")

    # 检查MySQL
    log_info("MySQL数据库: 请确认已执行 init_database.sql 初始化")

    return len(missing) == 0


# ============================================================
# Phase 1: 数据集制备
# ============================================================

def phase1_dataset():
    """数据集制备"""
    log_step(1, "酒店数据集全流程制备")

    # Step 1: 视频抽帧
    log_info("Step 1: 视频抽帧")
    raw_video_dir = PROJECT_ROOT / 'test_video' / 'hotel_raw'
    if list(raw_video_dir.glob('*')):
        success, _ = run_command(
            f'python extract_frames.py --input_dir test_video/hotel_raw '
            f'--output_dir dataset/det/hotel_img/images --interval 3'
        )
        if success:
            log_success("视频抽帧完成")
        else:
            log_warning("视频抽帧失败，请检查视频文件")
    else:
        log_warning("无原始视频文件，请将酒店监控录像放入 test_video/hotel_raw/")
        log_info("使用示例图片进行后续步骤...")
        _create_demo_images()

    # Step 2: 数据清洗
    log_info("Step 2: 三层数据清洗")
    img_dir = PROJECT_ROOT / 'dataset' / 'det' / 'hotel_img' / 'images'
    if list(img_dir.glob('*')):
        success, _ = run_command(
            f'python data_clean.py --input_dir dataset/det/hotel_img/images '
            f'--output_dir dataset/det/hotel_img'
        )
        if success:
            log_success("数据清洗完成")
        else:
            log_warning("数据清洗失败")
    else:
        log_warning("无待清洗图片")

    # Step 3: 数据标注(提示用户手动标注)
    log_info("Step 3: 数据标注")
    log_info("  使用LabelImg工具进行标注")
    log_info("  标注工具启动: labelimg dataset/det/hotel_img/images/ dataset/det/hotel_img/labels/")
    log_info("  标注规则: 仅单类别person, 遮挡<=40%完整框选")
    log_warning("  请完成标注后继续后续步骤")

    # 自动生成示例标签
    _create_demo_labels()

    # Step 4: 数据集划分
    log_info("Step 4: 数据集划分")
    success, _ = run_command(
        f'python split_dataset.py --source_img dataset/det/hotel_img/images '
        f'--source_label dataset/det/hotel_img/labels '
        f'--output dataset/det --yaml_path dataset/det/hotel_det.yaml'
    )
    if success:
        log_success("数据集划分完成")
    else:
        log_warning("数据集划分失败")

    # Step 5: 数据增强
    log_info("Step 5: 离线数据增强")
    clean_dir = PROJECT_ROOT / 'dataset' / 'det' / 'hotel_img' / 'images'
    if list(clean_dir.glob('*')):
        success, _ = run_command(
            f'python data_augment.py --input_dir dataset/det/hotel_img/images '
            f'--output_dir dataset/det/hotel_img/augmented --count 3'
        )
        if success:
            log_success("数据增强完成")
        else:
            log_warning("数据增强失败")
    else:
        log_warning("无待增强图片")


def _create_demo_images():
    """创建示例图片用于演示"""
    import numpy as np

    output_dir = PROJECT_ROOT / 'dataset' / 'det' / 'hotel_img' / 'images'
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from PIL import Image, ImageDraw
        use_pil = True
    except ImportError:
        use_pil = False

    for i in range(20):
        # 生成模拟图片(纯色背景+随机图案)
        img_array = np.random.randint(100, 200, (480, 640, 3), dtype=np.uint8)

        if use_pil:
            img = Image.fromarray(img_array)
            draw = ImageDraw.Draw(img)
            for _ in range(np.random.randint(1, 4)):
                x1 = np.random.randint(50, 550)
                y1 = np.random.randint(50, 400)
                x2 = x1 + np.random.randint(30, 80)
                y2 = y1 + np.random.randint(60, 150)
                color = tuple(np.random.randint(0, 100, 3).tolist())
                draw.rectangle([x1, y1, x2, y2], fill=color)
            img.save(str(output_dir / f'demo_{i:04d}.jpg'), quality=85)
        else:
            # 使用纯numpy保存(无图形库时生成BMP格式)
            try:
                import struct
                # 保存为PPM格式(最简单的图像格式)
                ppm_path = output_dir / f'demo_{i:04d}.ppm'
                h, w = img_array.shape[:2]
                with open(ppm_path, 'wb') as f:
                    f.write(f'P6\n{w} {h}\n255\n'.encode())
                    f.write(img_array.tobytes())
            except Exception:
                # 最后方案：保存原始二进制数据
                np.save(str(output_dir / f'demo_{i:04d}.npy'), img_array)

    log_info(f"已生成示例图片至: {output_dir}")


def _create_demo_labels():
    """创建示例标签文件"""
    import json

    label_dir = PROJECT_ROOT / 'dataset' / 'det' / 'hotel_img' / 'labels'
    label_dir.mkdir(parents=True, exist_ok=True)
    img_dir = PROJECT_ROOT / 'dataset' / 'det' / 'hotel_img' / 'images'

    for img_file in sorted(img_dir.glob('*.jpg')):
        # 生成简单的标签(假设图片中心有一个person)
        label_content = "0 0.5 0.5 0.3 0.4\n"
        label_path = label_dir / f"{img_file.stem}.txt"
        with open(label_path, 'w') as f:
            f.write(label_content)

    log_info(f"已生成 {len(list(label_dir.glob('*.txt')))} 个示例标签文件")


# ============================================================
# Phase 2: YOLOv8改进模型
# ============================================================

def phase2_yolo():
    """YOLOv8改进模型训练与消融实验"""
    log_step(2, "改进YOLOv8行人检测模型")

    # Step 1: 训练改进YOLOv8
    log_info("Step 1: 训练MixConv+BiFPN改进YOLOv8")
    config_yaml = PROJECT_ROOT / 'dataset' / 'det' / 'hotel_det.yaml'

    if config_yaml.exists():
        success, _ = run_command(
            f'python yolov8_exp/train_yolo.py --dataset_yaml dataset/det/hotel_det.yaml '
            f'--imgsz 640 --batch_size 16 --epochs 10 '
            f'--output_dir output/yolo_train_log'
        )
        if success:
            log_success("YOLOv8改进模型训练完成")
        else:
            log_warning("YOLOv8训练可能未完成(epoch设置较小用于演示)")
    else:
        log_warning("数据集配置文件不存在，请先执行Phase 1")

    # Step 2: 消融实验
    log_info("Step 2: 三组消融对照实验")
    success, _ = run_command(
        f'python yolov8_exp/ablation_experiment.py'
    )
    if success:
        log_success("消融实验完成")

    # Step 3: 测试
    log_info("Step 3: 模型测试与指标评估")
    model_path = PROJECT_ROOT / 'output' / 'yolo_train_log' / 'best.pt'
    if model_path.exists():
        success, _ = run_command(
            f'python yolov8_exp/test_yolo.py --model_path output/yolo_train_log/best.pt '
            f'--dataset_yaml dataset/det/hotel_det.yaml'
        )
        if success:
            log_success("模型测试完成")
    else:
        log_warning("训练权重不存在，跳过测试")


# ============================================================
# Phase 3: ResNet50重识别
# ============================================================

def phase3_reid():
    """ResNet50重识别模型训练"""
    log_step(3, "改进ResNet50行人重识别模型")

    # Step 1: 准备reid数据集
    reid_dir = PROJECT_ROOT / 'dataset' / 'reid' / 'hotel_reid'
    reid_dir.mkdir(parents=True, exist_ok=True)

    # 创建示例reid数据
    _create_demo_reid_data(reid_dir)

    # Step 2: 训练
    log_info("Step 2: 训练改进ResNet50重识别模型")
    success, _ = run_command(
        f'python resnet50_reid/train_reid.py --stage pretrain '
        f'--data_root dataset/reid/hotel_reid '
        f'--epochs 10 --batch_size 8 '
        f'--output_dir output/reid_train_log'
    )
    if success:
        log_success("ResNet50训练完成")
    else:
        log_warning("ResNet50训练可能未完成")

    # Step 3: 消融实验
    log_info("Step 3: 重识别消融实验")
    success, _ = run_command(f'python resnet50_reid/reid_ablation.py')
    if success:
        log_success("重识别消融实验完成")


def _create_demo_reid_data(reid_dir):
    """创建示例reid数据集"""
    import numpy as np

    try:
        from PIL import Image, ImageDraw
        use_pil = True
    except ImportError:
        use_pil = False

    # 创建train和test目录
    for split in ['train', 'test']:
        split_dir = reid_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)

        # 为每个人员ID创建示例
        for person_id in range(1, 11):
            person_id_str = f"{person_id:04d}"
            num_imgs = 8 if split == 'train' else 2

            for cam_id in range(1, 3):
                for frame_id in range(num_imgs):
                    # 生成模拟行人图片
                    img_array = np.random.randint(100, 200, (256, 128, 3), dtype=np.uint8)

                    if use_pil:
                        img = Image.fromarray(img_array)
                        draw = ImageDraw.Draw(img)
                        # 添加人像轮廓
                        cx, cy = 64, 100
                        draw.ellipse([cx - 20, cy - 65, cx + 20, cy - 15], fill=(80, 60, 50))
                        draw.rectangle([cx - 25, cy - 15, cx + 25, cy + 80], fill=(60, 70, 80))

                        filename = f"{person_id_str}_c{cam_id}s1_{frame_id:06d}_01.jpg"
                        img.save(str(split_dir / filename), quality=85)
                    else:
                        # 无PIL时保存npy格式
                        filename = f"{person_id_str}_c{cam_id}s1_{frame_id:06d}_01.npy"
                        np.save(str(split_dir / filename), img_array)

    log_info(f"已创建示例reid数据集: {reid_dir}")


# ============================================================
# Phase 4: Flask系统
# ============================================================

def phase4_system():
    """启动Flask系统"""
    log_step(4, "Flask+MySQL业务系统启动")

    log_info("启动Flask系统服务...")
    log_info("浏览器访问: http://localhost:5000")

    success, _ = run_command(
        f'python system_server/app.py --host 0.0.0.0 --port 5000 --debug',
        timeout=10
    )

    if not success:
        log_warning("Flask服务可能需要在独立终端启动")
        log_info("手动启动命令: python system_server/app.py --host 0.0.0.0 --port 5000")


# ============================================================
# Phase 5: 集成测试
# ============================================================

def phase5_integration_test():
    """全链路集成测试"""
    log_step(5, "全链路集成测试")

    log_info("测试项目结构完整性...")
    critical_files = [
        'init_database.sql',
        'extract_frames.py',
        'data_clean.py',
        'split_dataset.py',
        'data_augment.py',
        'yolov8_exp/custom_modules.py',
        'yolov8_exp/train_yolo.py',
        'yolov8_exp/test_yolo.py',
        'yolov8_exp/ablation_experiment.py',
        'resnet50_reid/model.py',
        'resnet50_reid/train_reid.py',
        'resnet50_reid/reid_ablation.py',
        'system_server/app.py',
        'system_server/db_mysql.py',
        'system_server/feature_extractor.py',
        'system_server/templates/dashboard.html',
        'system_server/templates/register.html',
        'system_server/templates/monitor.html',
        'system_server/templates/alerts.html',
        'system_server/templates/persons.html',
        'dataset/det/hotel_det.yaml',
        'requirements.txt',
    ]

    all_ok = True
    for f in critical_files:
        fpath = PROJECT_ROOT / f
        if fpath.exists():
            log_success(f"  ✓ {f}")
        else:
            log_error(f"  ✗ {f} 缺失")
            all_ok = False

    if all_ok:
        log_success("全部关键文件完整")

    # 生成测试报告
    report = {
        'test_date': datetime.now().isoformat(),
        'project_root': str(PROJECT_ROOT),
        'files_checked': len(critical_files),
        'all_present': all_ok,
        'directory_structure': {
            'dataset': {
                'det': ['coco_person', 'hotel_img', 'train', 'val', 'test'],
                'reid': ['market1501', 'hotel_reid']
            },
            'yolov8_exp': ['custom_modules.py', 'train_yolo.py', 'test_yolo.py', 'ablation_experiment.py'],
            'resnet50_reid': ['model.py', 'train_reid.py', 'reid_ablation.py'],
            'system_server': ['app.py', 'db_mysql.py', 'feature_extractor.py', 'templates/'],
            'test_video': ['hotel_raw/'],
            'output': ['yolo_train_log/', 'reid_train_log/', 'ablation_results/', 'alert_screenshots/']
        }
    }

    report_path = PROJECT_ROOT / 'output' / 'test_report.json'
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    log_success(f"测试报告已保存: {report_path}")


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="酒店异常人员实时监控识别系统 - 主运行脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python run_all.py --phase env_check      # 环境校验
  python run_all.py --phase dataset       # 数据集制备
  python run_all.py --phase yolo          # YOLOv8训练+消融
  python run_all.py --phase reid          # ResNet50训练+消融
  python run_all.py --phase system        # 启动Flask系统
  python run_all.py --phase test          # 集成测试
  python run_all.py --phase all           # 执行全部阶段
        """
    )

    parser.add_argument('--phase', type=str, default='all',
                        choices=['env_check', 'dataset', 'yolo', 'reid',
                                 'system', 'test', 'all'],
                        help='执行阶段')
    parser.add_argument('--skip_training', action='store_true',
                        help='跳过训练步骤(使用已有权重)')

    args = parser.parse_args()

    print(f"\n{Colors.MAGENTA}{'='*60}")
    print(f"  酒店异常人员实时监控识别系统")
    print(f"  执行阶段: {args.phase}")
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}{Colors.RESET}\n")

    start_time = time.time()

    if args.phase in ['env_check', 'all']:
        phase0_env_check()

    if args.phase in ['dataset', 'all']:
        phase1_dataset()

    if args.phase in ['yolo', 'all']:
        phase2_yolo()

    if args.phase in ['reid', 'all']:
        phase3_reid()

    if args.phase in ['system', 'all']:
        phase4_system()

    if args.phase in ['test', 'all']:
        phase5_integration_test()

    total_time = time.time() - start_time
    print(f"\n{Colors.GREEN}{'='*60}")
    print(f"  全部阶段执行完成!")
    print(f"  总耗时: {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"  项目路径: {PROJECT_ROOT}")
    print(f"{'='*60}{Colors.RESET}\n")

    # 输出后续操作指南
    print(f"{Colors.CYAN}后续操作:{Colors.RESET}")
    print("  1. 初始化数据库: mysql -u root -p < init_database.sql")
    print("  2. 标注数据集: labelimg dataset/det/hotel_img/images/")
    print("  3. 训练模型: python yolov8_exp/train_yolo.py")
    print("  4. 启动系统: python system_server/app.py")
    print("  5. 访问系统: http://localhost:5000\n")


if __name__ == '__main__':
    main()