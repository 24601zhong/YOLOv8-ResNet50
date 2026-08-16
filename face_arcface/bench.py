"""定位 ArcFace 训练瓶颈: 分段计时 backbone / head / loss / sync / dataloader"""
import os, sys, time
os.chdir(r'c:\D\Myproject\Data-processing\Hotel_Model_Train')
sys.path.insert(0, 'face_arcface')
import torch
from torch.amp import autocast, GradScaler
from iresnet import iresnet50
from arcface_loss import ArcFaceLayer, LabelSmoothingCrossEntropy

BS = 128
C = 10572

m = iresnet50(num_features=512).cuda()
a = ArcFaceLayer(512, C).cuda()
ce = LabelSmoothingCrossEntropy(0.1).cuda()
opt = torch.optim.SGD(list(m.parameters())+list(a.parameters()), lr=0.1, momentum=0.9)
scaler = GradScaler('cuda')
x = torch.randn(BS, 3, 112, 112).cuda()
y = torch.randint(0, C, (BS,)).cuda()

def timed(fn, n=10):
    for _ in range(3): fn()  # warmup
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n): fn()
    torch.cuda.synchronize()
    return (time.time()-t0)/n*1000  # ms

# 1. backbone forward only
def fwd_only():
    with autocast('cuda'):
        m(x)
print(f'backbone fwd         : {timed(fwd_only):7.1f} ms')

# 2. backbone fwd+bwd
def fwd_bwd():
    opt.zero_grad()
    with autocast('cuda'):
        f = m(x)
        loss = f.sum()
    loss.backward()
print(f'backbone fwd+bwd     : {timed(fwd_bwd):7.1f} ms')

# 3. full fwd+bwd (head+loss), no sync
def full():
    opt.zero_grad()
    with autocast('cuda'):
        f = m(x); logits = a(f, y); loss = ce(logits, y)
    scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
print(f'full fwd+bwd (no sync): {timed(full):7.1f} ms')

# 4. full + per-iter .item() syncs (as in train script)
def full_sync():
    opt.zero_grad()
    with autocast('cuda'):
        f = m(x); logits = a(f, y); loss = ce(logits, y)
    scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
    loss.item()
    (logits.argmax(dim=1) == y).sum().item()
print(f'full + 2 syncs       : {timed(full_sync):7.1f} ms')

# 5. head+loss only (no backbone)
def head_only():
    with torch.no_grad():
        f = torch.randn(BS, 512).cuda()
    opt.zero_grad()
    with autocast('cuda'):
        logits = a(f, y); loss = ce(logits, y)
    loss.backward()
print(f'head+loss only       : {timed(head_only):7.1f} ms')

# 6. fp16 vs fp32 head 对比
def full_no_amp():
    opt.zero_grad()
    f = m(x); logits = a(f, y); loss = ce(logits, y)
    loss.backward()
print(f'full NO-AMP (fp32)   : {timed(full_no_amp):7.1f} ms')

print('\n推断: 若 full ≈ backbone + head, 则无瓶颈; 若 full >> 两者之和, 说明 AMP/scale/sync 有开销')
