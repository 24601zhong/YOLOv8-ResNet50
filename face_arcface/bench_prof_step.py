"""用 torch profiler 剖分训练循环的 step, 找出 710ms 到底花在哪个算子"""
import os, sys, time
os.chdir(r'c:\D\Myproject\Data-processing\Hotel_Model_Train')
sys.path.insert(0, 'face_arcface')
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torch.amp import autocast, GradScaler
from iresnet import iresnet50
from arcface_loss import ArcFaceLayer, LabelSmoothingCrossEntropy

MEAN=[0.5,0.5,0.5]; STD=[0.5,0.5,0.5]


def main():
    tf = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(MEAN,STD)])
    ds = datasets.ImageFolder('dataset/face_casia/images', transform=tf)
    C = len(ds.classes)
    loader = DataLoader(ds, batch_size=128, shuffle=True, num_workers=4,
                        pin_memory=True, drop_last=True, persistent_workers=True)
    m = iresnet50(num_features=512).cuda()
    a = ArcFaceLayer(512, C, scale=30.0, margin=0.3).cuda()
    ce = LabelSmoothingCrossEntropy(0.1).cuda()
    opt = torch.optim.SGD([{'params': m.parameters()}, {'params': a.parameters()}],
                          lr=0.1, momentum=0.9, weight_decay=5e-4)
    scaler = GradScaler('cuda')
    m.train(); a.train()

    # 预热 (真实数据)
    it = iter(loader)
    for _ in range(8):
        imgs, labels = next(it)
        imgs = imgs.cuda(non_blocking=True); labels = labels.cuda(non_blocking=True)
        opt.zero_grad()
        with autocast('cuda'):
            f = m(imgs); logits = a(f, labels); loss = ce(logits, labels)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
    torch.cuda.synchronize()

    # 单独测 step 耗时 (真实数据, 同步隔离)
    imgs, labels = next(it)
    imgs = imgs.cuda(non_blocking=True); labels = labels.cuda(non_blocking=True)
    opt.zero_grad()
    with autocast('cuda'):
        f = m(imgs); logits = a(f, labels); loss = ce(logits, labels)
    scaler.scale(loss).backward()
    torch.cuda.synchronize()
    s = time.time(); scaler.step(opt); scaler.update(); torch.cuda.synchronize()
    print(f'[real-data] step+sync = {(time.time()-s)*1000:.1f} ms', flush=True)

    # profiler 剖分 fwd+bwd+step 各算子 CUDA 时间
    imgs, labels = next(it)
    imgs = imgs.cuda(non_blocking=True); labels = labels.cuda(non_blocking=True)
    opt.zero_grad()
    with torch.autograd.profiler.profile(use_cuda=True, record_shapes=False) as prof:
        with autocast('cuda'):
            f = m(imgs); logits = a(f, labels); loss = ce(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update()
        torch.cuda.synchronize()
    print('\n===== CUDA time 前 25 =====')
    print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=25))


if __name__ == '__main__':
    main()
