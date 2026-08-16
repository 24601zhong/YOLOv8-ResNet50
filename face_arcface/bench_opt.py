"""隔离 optimizer 配置对 step 速度的影响: param groups vs weight_decay"""
import os, sys, time
os.chdir(r'c:\D\Myproject\Data-processing\Hotel_Model_Train')
sys.path.insert(0, 'face_arcface')
import torch
from torch.amp import autocast, GradScaler
from iresnet import iresnet50
from arcface_loss import ArcFaceLayer, LabelSmoothingCrossEntropy

BS=128; C=10572

def build_opt(m, a, mode):
    if mode == 'flat':
        return torch.optim.SGD(list(m.parameters())+list(a.parameters()), lr=0.1, momentum=0.9)
    if mode == 'flat_wd':
        return torch.optim.SGD(list(m.parameters())+list(a.parameters()), lr=0.1, momentum=0.9, weight_decay=5e-4)
    if mode == 'groups':
        return torch.optim.SGD([{'params': m.parameters()}, {'params': a.parameters()}], lr=0.1, momentum=0.9)
    if mode == 'groups_wd':
        return torch.optim.SGD([{'params': m.parameters()}, {'params': a.parameters()}], lr=0.1, momentum=0.9, weight_decay=5e-4)

def run(mode, n=20):
    m = iresnet50(num_features=512).cuda()
    a = ArcFaceLayer(512, C).cuda()
    ce = LabelSmoothingCrossEntropy(0.1).cuda()
    opt = build_opt(m, a, mode)
    scaler = GradScaler('cuda')
    x = torch.randn(BS, 3, 112, 112).cuda()
    y = torch.randint(0, C, (BS,)).cuda()
    # warmup
    for _ in range(3):
        opt.zero_grad()
        with autocast('cuda'):
            f = m(x); logits = a(f, y); loss = ce(logits, y)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
    torch.cuda.synchronize()
    # time full loop
    t0 = time.time()
    for _ in range(n):
        opt.zero_grad()
        with autocast('cuda'):
            f = m(x); logits = a(f, y); loss = ce(logits, y)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
    torch.cuda.synchronize()
    full = (time.time()-t0)/n*1000
    del m, a, ce, opt, scaler
    torch.cuda.empty_cache()
    print(f'[{mode}]  full={full:6.1f} ms/iter')
    return full

# 单独测 step 耗时
def step_only(mode, n=20):
    m = iresnet50(num_features=512).cuda()
    a = ArcFaceLayer(512, C).cuda()
    ce = LabelSmoothingCrossEntropy(0.1).cuda()
    opt = build_opt(m, a, mode)
    scaler = GradScaler('cuda')
    x = torch.randn(BS, 3, 112, 112).cuda()
    y = torch.randint(0, C, (BS,)).cuda()
    for _ in range(3):
        opt.zero_grad()
        with autocast('cuda'):
            f = m(x); logits = a(f, y); loss = ce(logits, y)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
    torch.cuda.synchronize()
    t_step = 0.0
    for _ in range(n):
        opt.zero_grad()
        with autocast('cuda'):
            f = m(x); logits = a(f, y); loss = ce(logits, y)
        scaler.scale(loss).backward()
        torch.cuda.synchronize()
        s = time.time(); scaler.step(opt); scaler.update(); torch.cuda.synchronize(); t_step += time.time()-s
    del m, a, ce, opt, scaler
    torch.cuda.empty_cache()
    print(f'[{mode}]  step+sync={t_step/n*1000:6.1f} ms')
    return t_step/n*1000

if __name__ == '__main__':
    for mode in ['flat', 'flat_wd', 'groups', 'groups_wd']:
        run(mode)
    print('--- step 单独耗时 ---')
    for mode in ['flat', 'flat_wd', 'groups', 'groups_wd']:
        step_only(mode)
