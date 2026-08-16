"""精确复刻训练循环, 拆解 step 内部 (unscale vs opt.step vs update), 看 693ms 从哪来"""
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

    def one_epoch(tag, n, mode):
        m.train(); a.train()
        p_unscale = p_optstep = p_update = p_lossitem = 0.0
        t_ep = time.time()
        for i, (imgs, labels) in enumerate(loader):
            if i >= n: break
            imgs = imgs.cuda(non_blocking=True); labels = labels.cuda(non_blocking=True)
            opt.zero_grad()
            with autocast('cuda'):
                f = m(imgs); logits = a(f, labels); loss = ce(logits, labels)
            scaler.scale(loss).backward()
            torch.cuda.synchronize()          # 复刻 PROFILE: step 前 sync, 隔离 fwd+bwd
            s = time.time(); scaler.unscale_(opt); torch.cuda.synchronize(); p_unscale += time.time()-s
            s = time.time(); opt.step(); torch.cuda.synchronize(); p_optstep += time.time()-s
            s = time.time(); scaler.update(); p_update += time.time()-s
            s = time.time(); _ = loss.item(); p_lossitem += time.time()-s
        wall = (time.time()-t_ep)/n*1000
        print(f'[{tag}] n={n} wall={wall:6.1f}  unscale={p_unscale/n*1000:6.1f} '
              f'optstep={p_optstep/n*1000:6.1f} update={p_update/n*1000:6.1f} '
              f'lossitem={p_lossitem/n*1000:6.1f}  (ms/iter)', flush=True)

    # 先跑一个 epoch 预热 (80 iter, 复刻真实训练量)
    one_epoch('epoch0_warm', 80, 'warm')
    # 复刻训练脚本: epoch 结束 torch.save(含 optimizer state_dict)
    ckpt = {'backbone': m.state_dict(), 'arcface': a.state_dict(),
            'optimizer': opt.state_dict(), 'epoch': 0, 'num_classes': C}
    torch.save(ckpt, 'face_arcface/output_timing_test/_bench_last.pt')
    print('  [torch.save done]', flush=True)
    # 再测第二个 epoch (稳态)
    one_epoch('epoch1_after_save', 80, 'steady')


if __name__ == '__main__':
    main()
