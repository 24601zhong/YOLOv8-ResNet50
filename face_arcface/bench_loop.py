"""真实训练循环计时 (含 if __name__=='__main__' 守卫)"""
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

def run(workers, use_sync, n=20):
    tf = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(MEAN,STD)])
    ds = datasets.ImageFolder('dataset/face_casia/images', transform=tf)
    C = len(ds.classes)
    loader = DataLoader(ds, batch_size=128, shuffle=True, num_workers=workers,
                        pin_memory=True, drop_last=True, persistent_workers=(workers>0))
    m = iresnet50(num_features=512).cuda()
    a = ArcFaceLayer(512, C).cuda()
    ce = LabelSmoothingCrossEntropy(0.1).cuda()
    opt = torch.optim.SGD(list(m.parameters())+list(a.parameters()), lr=0.1, momentum=0.9)
    scaler = GradScaler('cuda')
    it = iter(loader)
    for _ in range(3):
        next(it)
    torch.cuda.synchronize()
    t0 = time.time()
    for i in range(n):
        imgs, labels = next(it)
        imgs = imgs.cuda(non_blocking=True); labels = labels.cuda(non_blocking=True)
        opt.zero_grad()
        with autocast('cuda'):
            f = m(imgs); logits = a(f, labels); loss = ce(logits, labels)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        if use_sync:
            loss.item(); (logits.argmax(dim=1)==labels).sum().item()
    torch.cuda.synchronize()
    wall = (time.time()-t0)/n*1000
    print(f'workers={workers} sync={use_sync}:  wall={wall:6.1f} ms/iter  ({128/wall*1000:.0f} img/s)')
    del loader, m, a, ce, opt, scaler, ds
    torch.cuda.empty_cache()

if __name__ == '__main__':
    run(4, True)    # 当前配置 (复现 860ms)
    run(4, False)   # 去掉 per-iter .item() sync
    run(8, True)    # 更多 workers
