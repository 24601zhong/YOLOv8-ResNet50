"""
============================================================
ArcFace 人脸识别训练 (Kaggle 版, 可移植, 无 Windows 硬编码路径)
IResNet50 + ArcFace, 数据集 CASIA-WebFace

Kaggle 上数据是原始 .rec (train.rec + train.idx), 需先转换:
  kaggle_convert.py 把 .rec → /kaggle/working/dataset/face_casia/images (ImageFolder)
  验证 .bin 在 /kaggle/input/datasets/debarghamitraroy/casia-webface/eval

iresnet.py / arcface_loss.py 与本脚本同目录 (notebook 里用 %%writefile 生成)

用法:
  python kaggle_train.py --epochs 20 --batch 256
  python kaggle_train.py --epochs 1 --batch 64 --max-steps 3   # 冒烟

性能: torch.backends.cudnn.benchmark=True (固定 112×112 下 autotune,
      否则 conv 反向 dgrad 慢 4 倍)
============================================================
"""
import os
import sys
import math
import time
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torch.amp import autocast, GradScaler
from iresnet import iresnet50
from arcface_loss import ArcFaceLayer, LabelSmoothingCrossEntropy

# 转换后的 ImageFolder 输出目录 (kaggle_convert.py 的默认输出)
DEFAULT_DATA = '/kaggle/working/dataset/face_casia/images'


def arg(name, default):
    try:
        i = sys.argv.index(name)
        return sys.argv[i + 1]
    except (ValueError, IndexError):
        return default


def main():
    # 固定输入尺寸(112×112)下让 cuDNN autotune 选最优卷积算法
    torch.backends.cudnn.benchmark = True

    EPOCHS = int(arg('--epochs', 20))
    BATCH = int(arg('--batch', 256))     # T4 16GB 可用 256; OOM 则改 128
    WORKERS = int(arg('--workers', 2))   # Kaggle 一般 4 核 CPU
    LR0 = float(arg('--lr0', 0.1))
    WARMUP = int(arg('--warmup', 2))
    DATA = arg('--data', DEFAULT_DATA)
    OUT_DIR = arg('--out', os.path.join(SCRIPT_DIR, 'output'))
    RESUME = arg('--resume', '')
    MAX_STEPS = int(arg('--max-steps', 0))  # >0 时每 epoch 只跑 N 个 batch (冒烟)

    os.makedirs(OUT_DIR, exist_ok=True)

    print('=' * 64)
    print(f'  ArcFace Face Recognition Training (Kaggle)')
    print(f'  Start: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print('=' * 64)
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f'  GPU: {gpu} ({vram:.1f} GB VRAM)  CUDA {torch.version.cuda}  '
              f'cudnn.benchmark={torch.backends.cudnn.benchmark}')
    else:
        print('  [ERROR] CUDA not available'); sys.exit(1)

    # ---------------- 数据 ----------------
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    dataset = datasets.ImageFolder(DATA, transform=transform)
    num_classes = len(dataset.classes)
    loader = DataLoader(dataset, batch_size=BATCH, shuffle=True,
                        num_workers=WORKERS, pin_memory=True, drop_last=True,
                        persistent_workers=(WORKERS > 0))
    iters_per_epoch = len(loader)
    print(f'  数据: {DATA}')
    print(f'  样本数: {len(dataset)}  身份数: {num_classes}  iters/epoch: {iters_per_epoch}')
    print(f'  配置: epochs={EPOCHS} batch={BATCH} workers={WORKERS} lr0={LR0} warmup={WARMUP}')

    # ---------------- 模型 ----------------
    backbone = iresnet50(num_features=512).cuda()
    arcface = ArcFaceLayer(512, num_classes, scale=30.0, margin=0.3).cuda()
    ce_loss = LabelSmoothingCrossEntropy(epsilon=0.1).cuda()

    n_params = sum(p.numel() for p in backbone.parameters())
    print(f'  Backbone IResNet50 参数量: {n_params/1e6:.2f} M')

    # ---------------- 优化器 / 调度 ----------------
    optimizer = torch.optim.SGD(
        [{'params': backbone.parameters()},
         {'params': arcface.parameters()}],
        lr=LR0, momentum=0.9, weight_decay=5e-4)
    scaler = GradScaler('cuda')

    start_epoch = 0
    if RESUME and os.path.exists(RESUME):
        ckpt = torch.load(RESUME, map_location='cpu', weights_only=False)
        backbone.load_state_dict(ckpt['backbone'])
        arcface.load_state_dict(ckpt['arcface'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
        print(f'  [resume] 从 {RESUME} 续训, 从 epoch {start_epoch} 开始')

    def lr_at(epoch):
        if epoch < WARMUP:
            return LR0 * (epoch + 1) / WARMUP
        progress = (epoch - WARMUP) / max(1, EPOCHS - WARMUP)
        return LR0 * 0.01 + 0.5 * (LR0 - LR0 * 0.01) * (1 + math.cos(math.pi * progress))

    # ---------------- 训练循环 ----------------
    print(f'\n  {"Epoch":>6s} | {"lr":>9s} | {"loss":>8s} | {"acc":>7s} | {"time":>8s} | {"剩余":>8s}')
    print('  ' + '-' * 60)
    t_start = time.time()

    for epoch in range(start_epoch, EPOCHS):
        lr = lr_at(epoch)
        for g in optimizer.param_groups:
            g['lr'] = lr

        backbone.train()
        arcface.train()
        run_loss = 0.0
        run_correct = 0
        run_total = 0
        t_ep = time.time()

        for i, (imgs, labels) in enumerate(loader):
            if MAX_STEPS and i >= MAX_STEPS:
                break
            imgs = imgs.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            optimizer.zero_grad()
            with autocast('cuda'):
                feats = backbone(imgs)
                logits = arcface(feats, labels)
                loss = ce_loss(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            run_loss += loss.item() * imgs.size(0)
            run_correct += (logits.argmax(dim=1) == labels).sum().item()
            run_total += imgs.size(0)

        n_it = max(1, i + 1)
        avg_loss = run_loss / run_total
        acc = run_correct / run_total
        el = time.time() - t_ep
        total_el = time.time() - t_start
        remain = total_el / (epoch - start_epoch + 1) * (EPOCHS - epoch - 1)
        print(f'  {epoch+1:6d} | {lr:9.6f} | {avg_loss:8.4f} | {acc:7.4f} | {el:7.1f}s | {remain/3600:6.1f}h')

        # 存档 (每 epoch 存 last.pt, 每 5 epoch 存 epoch_N.pt)
        ckpt = {
            'backbone': backbone.state_dict(),
            'arcface': arcface.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'num_classes': num_classes,
        }
        torch.save(ckpt, os.path.join(OUT_DIR, 'last.pt'))
        if (epoch + 1) % 5 == 0:
            torch.save(ckpt, os.path.join(OUT_DIR, f'epoch_{epoch+1}.pt'))

    print('=' * 64)
    print(f'  Training Complete!  End: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'  模型保存在: {OUT_DIR}/last.pt')
    print('=' * 64)


if __name__ == '__main__':
    main()
