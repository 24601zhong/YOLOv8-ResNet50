"""完全复刻 train_face_arcface.py 的 main() 逐阶段计时, 定位 515ms 差额"""
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
LOG_PATH = os.path.join(PROJECT, 'face_arcface/bench_exact.log')

class Tee:
    def __init__(self, *files): self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj); f.flush()
    def flush(self):
        for f in self.files: f.flush()

def main():
    log_file = open(LOG_PATH, 'a', buffering=1)
    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = sys.stdout

    EPOCHS=20; BATCH=128; WORKERS=4; LR0=0.1; WARMUP=2
    DATA='dataset/face_casia/images'; MAX_STEPS=40
    transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize([0.5]*3,[0.5]*3)])
    dataset = datasets.ImageFolder(DATA, transform=transform)
    num_classes = len(dataset.classes)
    loader = DataLoader(dataset, batch_size=BATCH, shuffle=True, num_workers=WORKERS, pin_memory=True, drop_last=True, persistent_workers=(WORKERS>0))
    backbone = iresnet50(num_features=512).cuda()
    arcface = ArcFaceLayer(512, num_classes, scale=30.0, margin=0.3).cuda()
    ce_loss = LabelSmoothingCrossEntropy(epsilon=0.1).cuda()
    optimizer = torch.optim.SGD([{'params': backbone.parameters()}, {'params': arcface.parameters()}], lr=LR0, momentum=0.9, weight_decay=5e-4)
    scaler = GradScaler('cuda')

    def lr_at(epoch):
        if epoch < WARMUP: return LR0*(epoch+1)/WARMUP
        prog = (epoch-WARMUP)/max(1,EPOCHS-WARMUP)
        return LR0*0.01 + 0.5*(LR0-LR0*0.01)*(1+math.cos(math.pi*prog))

    for epoch in range(EPOCHS):
        lr = lr_at(epoch)
        for g in optimizer.param_groups: g['lr'] = lr
        backbone.train(); arcface.train()
        run_loss=0.0; run_correct=0; run_total=0
        t_ep = time.time()
        t_next=t_cuda=t_zero=t_fwd=t_bwd=t_step=t_sync=0.0
        for i, (imgs, labels) in enumerate(loader):
            if MAX_STEPS and i >= MAX_STEPS: break
            s=time.time()
            imgs = imgs.cuda(non_blocking=True); labels = labels.cuda(non_blocking=True); t_cuda+=time.time()-s
            s=time.time(); optimizer.zero_grad(); t_zero+=time.time()-s
            s=time.time()
            with autocast('cuda'):
                feats = backbone(imgs); logits = arcface(feats, labels); loss = ce_loss(logits, labels)
            t_fwd+=time.time()-s
            s=time.time(); scaler.scale(loss).backward(); t_bwd+=time.time()-s
            s=time.time(); scaler.step(optimizer); scaler.update(); t_step+=time.time()-s
            s=time.time(); run_loss += loss.item()*imgs.size(0); run_correct += (logits.argmax(dim=1)==labels).sum().item(); run_total += imgs.size(0); t_sync+=time.time()-s
        el = time.time()-t_ep
        n = min(MAX_STEPS, run_total//BATCH) or run_total//BATCH
        print(f'  elapsed={el:.1f}s for {MAX_STEPS} iters = {el/MAX_STEPS*1000:.1f} ms/iter | '
              f'cuda={t_cuda/MAX_STEPS*1000:4.1f} zero={t_zero/MAX_STEPS*1000:4.1f} fwd={t_fwd/MAX_STEPS*1000:6.1f} '
              f'bwd={t_bwd/MAX_STEPS*1000:6.1f} step={t_step/MAX_STEPS*1000:5.1f} sync={t_sync/MAX_STEPS*1000:5.1f}')
        break  # 只跑一个 epoch
    log_file.close()

if __name__ == '__main__':
    main()
