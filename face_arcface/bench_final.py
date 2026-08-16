"""最终 A/B: flat vs groups_wd, 都用 DataLoader, 同进程对照"""
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

def make_loader():
    tf = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(MEAN,STD)])
    ds = datasets.ImageFolder('dataset/face_casia/images', transform=tf)
    return DataLoader(ds, batch_size=128, shuffle=True, num_workers=4,
                      pin_memory=True, drop_last=True, persistent_workers=True)

def run(tag, mode, n=30):
    loader = make_loader()
    C = len(loader.dataset.classes)
    m = iresnet50(num_features=512).cuda()
    a = ArcFaceLayer(512, C, scale=30.0, margin=0.3).cuda()
    ce = LabelSmoothingCrossEntropy(0.1).cuda()
    if mode == 'flat':
        opt = torch.optim.SGD(list(m.parameters())+list(a.parameters()), lr=0.1, momentum=0.9)
    else:
        opt = torch.optim.SGD([{'params': m.parameters()}, {'params': a.parameters()}],
                              lr=0.1, momentum=0.9, weight_decay=5e-4)
    scaler = GradScaler('cuda')
    it = iter(loader)
    for _ in range(3):  # warmup
        imgs, labels = next(it)
        imgs = imgs.cuda(non_blocking=True); labels = labels.cuda(non_blocking=True)
        opt.zero_grad()
        with autocast('cuda'):
            f = m(imgs); logits = a(f, labels); loss = ce(logits, labels)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
    torch.cuda.synchronize()
    t0 = time.time()
    for i in range(n):
        imgs, labels = next(it)
        imgs = imgs.cuda(non_blocking=True); labels = labels.cuda(non_blocking=True)
        opt.zero_grad()
        with autocast('cuda'):
            f = m(imgs); logits = a(f, labels); loss = ce(logits, labels)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        loss.item(); (logits.argmax(dim=1) == labels).sum().item()
    torch.cuda.synchronize()
    wall = (time.time() - t0) / n * 1000
    print(f'[{tag}] mode={mode}:  {wall:6.1f} ms/iter  ({128/wall*1000:.0f} img/s)')
    del loader, m, a, ce, opt, scaler
    torch.cuda.empty_cache()

if __name__ == '__main__':
    run('P', 'flat', n=30)
    run('Q', 'groups_wd', n=30)
    run('R', 'flat', n=30)
    run('S', 'groups_wd', n=30)
