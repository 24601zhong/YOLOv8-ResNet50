"""测 per-iter 是否随迭代数退化 (300 步分块计时)"""
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
    a = ArcFaceLayer(512, C).cuda()
    ce = LabelSmoothingCrossEntropy(0.1).cuda()
    opt = torch.optim.SGD(list(m.parameters())+list(a.parameters()), lr=0.1, momentum=0.9)
    scaler = GradScaler('cuda')
    it = iter(loader)
    N = 300
    chunk = 50
    t_prev = time.time()
    torch.cuda.synchronize(); t0 = time.time()
    for i in range(N):
        imgs, labels = next(it)
        imgs = imgs.cuda(non_blocking=True); labels = labels.cuda(non_blocking=True)
        opt.zero_grad()
        with autocast('cuda'):
            f = m(imgs); logits = a(f, labels); loss = ce(logits, labels)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        loss.item(); (logits.argmax(dim=1)==labels).sum().item()
        if (i+1) % chunk == 0:
            torch.cuda.synchronize()
            now = time.time()
            print(f'  iters {i+1-chunk:3d}-{i+1:3d}: {(now-t_prev)/chunk*1000:6.1f} ms/iter   (累计均 {(now-t0)/(i+1)*1000:6.1f} ms)')
            t_prev = now
    vram = torch.cuda.max_memory_allocated()/1e9
    print(f'  peak VRAM: {vram:.2f} GB')

if __name__ == '__main__':
    main()
