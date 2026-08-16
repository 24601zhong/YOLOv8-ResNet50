"""逐阶段计时, 复刻 train_face_arcface.py 的精确 setup, 定位 907ms vs 307ms 差异"""
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

def main(weight_decay, n=30):
    tf = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(MEAN,STD)])
    ds = datasets.ImageFolder('dataset/face_casia/images', transform=tf)
    C = len(ds.classes)
    loader = DataLoader(ds, batch_size=128, shuffle=True, num_workers=4,
                        pin_memory=True, drop_last=True, persistent_workers=True)
    backbone = iresnet50(num_features=512).cuda()
    arcface = ArcFaceLayer(512, C, scale=30.0, margin=0.3).cuda()
    ce_loss = LabelSmoothingCrossEntropy(0.1).cuda()
    opt = torch.optim.SGD(
        [{'params': backbone.parameters()}, {'params': arcface.parameters()}],
        lr=0.1, momentum=0.9, weight_decay=weight_decay)
    scaler = GradScaler('cuda')
    it = iter(loader)
    for _ in range(3): next(it)
    torch.cuda.synchronize()

    t_next = t_cuda = t_zero = t_fwd = t_bwd = t_step = t_sync = 0.0
    for i in range(n):
        s = time.time(); imgs, labels = next(it); t_next += time.time()-s
        s = time.time(); imgs = imgs.cuda(non_blocking=True); labels = labels.cuda(non_blocking=True); t_cuda += time.time()-s
        s = time.time(); opt.zero_grad(); t_zero += time.time()-s
        s = time.time()
        with autocast('cuda'):
            f = backbone(imgs); logits = arcface(f, labels); loss = ce_loss(logits, labels)
        t_fwd += time.time()-s
        s = time.time(); scaler.scale(loss).backward(); t_bwd += time.time()-s
        s = time.time(); scaler.step(opt); scaler.update(); t_step += time.time()-s
        s = time.time(); loss.item(); (logits.argmax(dim=1)==labels).sum().item(); t_sync += time.time()-s
    torch.cuda.synchronize()
    print(f'weight_decay={weight_decay}:  '
          f'next={t_next/n*1000:5.1f}  cuda={t_cuda/n*1000:4.1f}  zero={t_zero/n*1000:4.1f}  '
          f'fwd={t_fwd/n*1000:6.1f}  bwd={t_bwd/n*1000:6.1f}  step={t_step/n*1000:5.1f}  sync={t_sync/n*1000:5.1f}  (ms/iter)')

if __name__ == '__main__':
    main(weight_decay=5e-4)   # 训练脚本配置
    main(weight_decay=0.0)    # benchmark 配置
