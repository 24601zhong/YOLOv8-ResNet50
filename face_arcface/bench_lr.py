"""测试: 修改 optimizer.param_groups 的 lr 是否导致 SGD 回退慢路径"""
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

def build():
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
    return loader, m, a, ce, opt, scaler

def run(tag, change_lr, n=30):
    loader, m, a, ce, opt, scaler = build()
    m.train(); a.train()
    if change_lr:
        for g in opt.param_groups:
            g['lr'] = 0.05  # 复刻训练脚本 warmup 阶段改 lr
    it = iter(loader)
    for _ in range(8):
        imgs, labels = next(it)
        imgs = imgs.cuda(non_blocking=True); labels = labels.cuda(non_blocking=True)
        opt.zero_grad()
        with autocast('cuda'):
            f = m(imgs); logits = a(f, labels); loss = ce(logits, labels)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
    torch.cuda.synchronize()
    t0 = time.time()
    t_gpu = 0.0
    for i in range(n):
        imgs, labels = next(it)
        imgs = imgs.cuda(non_blocking=True); labels = labels.cuda(non_blocking=True)
        opt.zero_grad()
        with autocast('cuda'):
            f = m(imgs); logits = a(f, labels); loss = ce(logits, labels)
        scaler.scale(loss).backward()
        s = time.time(); torch.cuda.synchronize(); t_gpu += time.time() - s
        scaler.step(opt); scaler.update()
        loss.item(); (logits.argmax(dim=1) == labels).sum().item()
    torch.cuda.synchronize()
    wall = (time.time() - t0) / n * 1000
    print(f'[{tag}] change_lr={int(change_lr)}:  wall={wall:6.1f}  fwd+bwd_GPU={t_gpu/n*1000:6.1f}  (ms/iter)')
    del loader, m, a, ce, opt, scaler
    torch.cuda.empty_cache()

if __name__ == '__main__':
    run('A', change_lr=False)
    run('B', change_lr=True)
