"""带预热的最终二分: Tee × enumerate 谁是稳态 912ms 的真凶"""
import os, sys, time, math
os.chdir(r'c:\D\Myproject\Data-processing\Hotel_Model_Train')
sys.path.insert(0, 'face_arcface')
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torch.amp import autocast, GradScaler
from iresnet import iresnet50
from arcface_loss import ArcFaceLayer, LabelSmoothingCrossEntropy

PROJECT = r'c:\D\Myproject\Data-processing\Hotel_Model_Train'
LOG_PATH = os.path.join(PROJECT, 'face_arcface/bench_bisect2.log')

class Tee:
    def __init__(self, *files): self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj); f.flush()
    def flush(self):
        for f in self.files: f.flush()

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

def step(imgs, labels, m, a, ce, opt, scaler):
    imgs = imgs.cuda(non_blocking=True); labels = labels.cuda(non_blocking=True)
    opt.zero_grad()
    with autocast('cuda'):
        f = m(imgs); logits = a(f, labels); loss = ce(logits, labels)
    scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
    loss.item(); (logits.argmax(dim=1) == labels).sum().item()

def run(tag, use_tee, use_enumerate, n=40, warm=8):
    if use_tee:
        log_file = open(LOG_PATH, 'a', buffering=1)
        sys.stdout = Tee(sys.stdout, log_file)
        sys.stderr = sys.stdout
    loader, m, a, ce, opt, scaler = build()
    m.train(); a.train()
    # 预热 warm 步
    it = iter(loader)
    for _ in range(warm):
        imgs, labels = next(it)
        step(imgs, labels, m, a, ce, opt, scaler)
    torch.cuda.synchronize()
    t0 = time.time()
    if use_enumerate:
        # 用 enumerate(loader) 新建迭代器 (复刻训练脚本每 epoch 新建)
        for i, (imgs, labels) in enumerate(loader):
            if i >= n: break
            step(imgs, labels, m, a, ce, opt, scaler)
    else:
        it2 = iter(loader)
        for i in range(n):
            imgs, labels = next(it2)
            step(imgs, labels, m, a, ce, opt, scaler)
    torch.cuda.synchronize()
    wall = (time.time() - t0) / n * 1000
    del loader, m, a, ce, opt, scaler
    torch.cuda.empty_cache()
    if use_tee:
        sys.stdout = sys.__stdout__; sys.stderr = sys.__stderr__
    print(f'[{tag}] tee={int(use_tee)} enum={int(use_enumerate)}:  {wall:6.1f} ms/iter')

if __name__ == '__main__':
    run('A', use_tee=False, use_enumerate=False)  # 基线 (应≈345)
    run('B', use_tee=True,  use_enumerate=False)  # +Tee
    run('C', use_tee=False, use_enumerate=True)   # +enumerate
    run('D', use_tee=True,  use_enumerate=True)   # 训练脚本完整复刻
