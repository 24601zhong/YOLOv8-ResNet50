"""同进程 A/B: 训练式循环(enumerate+break) vs bench式循环(next), 测 fwd+bwd GPU 时间"""
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

def body(imgs, labels, m, a, ce, opt, scaler, measure):
    imgs = imgs.cuda(non_blocking=True); labels = labels.cuda(non_blocking=True)
    opt.zero_grad()
    with autocast('cuda'):
        f = m(imgs); logits = a(f, labels); loss = ce(logits, labels)
    scaler.scale(loss).backward()
    if measure:
        torch.cuda.synchronize()
        t_gpu = time.time()
        # 已在 sync 后, 无等待
        t_gpu = None  # 不这样测; 见下
    scaler.step(opt); scaler.update()
    loss.item(); (logits.argmax(dim=1) == labels).sum().item()

def run(tag, use_enumerate, max_steps, n=40):
    loader, m, a, ce, opt, scaler = build()
    m.train(); a.train()
    # 预热 10 步 (用 next)
    it = iter(loader)
    for _ in range(10):
        imgs, labels = next(it)
        body(imgs, labels, m, a, ce, opt, scaler, measure=False)
    torch.cuda.synchronize()
    # 计时: 显式测 fwd+bwd GPU 时间
    t_wall0 = time.time()
    t_fwdbwd = 0.0
    t_fwd_enq = t_bwd_enq = 0.0
    cnt = 0
    if use_enumerate:
        for i, (imgs, labels) in enumerate(loader):
            if i >= max_steps: break
            imgs = imgs.cuda(non_blocking=True); labels = labels.cuda(non_blocking=True)
            opt.zero_grad()
            s = time.time()
            with autocast('cuda'):
                f = m(imgs); logits = a(f, labels); loss = ce(logits, labels)
            t_fwd_enq += time.time() - s
            s = time.time(); scaler.scale(loss).backward(); t_bwd_enq += time.time() - s
            s = time.time(); torch.cuda.synchronize(); t_fwdbwd += time.time() - s
            scaler.step(opt); scaler.update()
            loss.item(); (logits.argmax(dim=1) == labels).sum().item()
            cnt += 1
    else:
        it2 = iter(loader)
        for i in range(max_steps):
            imgs, labels = next(it2)
            imgs = imgs.cuda(non_blocking=True); labels = labels.cuda(non_blocking=True)
            opt.zero_grad()
            s = time.time()
            with autocast('cuda'):
                f = m(imgs); logits = a(f, labels); loss = ce(logits, labels)
            t_fwd_enq += time.time() - s
            s = time.time(); scaler.scale(loss).backward(); t_bwd_enq += time.time() - s
            s = time.time(); torch.cuda.synchronize(); t_fwdbwd += time.time() - s
            scaler.step(opt); scaler.update()
            loss.item(); (logits.argmax(dim=1) == labels).sum().item()
            cnt += 1
    torch.cuda.synchronize()
    wall = (time.time() - t_wall0) / cnt * 1000
    print(f'[{tag}] enum={int(use_enumerate)} n={cnt}:  wall={wall:6.1f}  '
          f'fwd_enq={t_fwd_enq/cnt*1000:5.1f}  bwd_enq={t_bwd_enq/cnt*1000:5.1f}  '
          f'fwd+bwd_GPU={t_fwdbwd/cnt*1000:6.1f}  (ms/iter)')
    del loader, m, a, ce, opt, scaler
    torch.cuda.empty_cache()

if __name__ == '__main__':
    run('A', use_enumerate=False, max_steps=40)  # bench式 (next)
    run('B', use_enumerate=True,  max_steps=40)  # 训练式 (enumerate)
