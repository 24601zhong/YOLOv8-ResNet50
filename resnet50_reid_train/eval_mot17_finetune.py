"""MOT17 微调模型评估 — 消除 batch-size 偏差的完整 P/R/F1 报告"""
import sys, os
os.chdir(r'c:\D\Myproject\Data-processing\Hotel_Model_Train')
sys.path.insert(0, 'resnet50_reid_train')

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
from tqdm import tqdm
from model import create_model
from train_reid import ReIDDataset, get_test_transforms

DEVICE = torch.device('cuda:0')
CKPT = Path('resnet50_reid_train/train_output/mot17_finetune_log/best_mot17_finetune.pth')
TRAIN_DIR = Path('dataset/mot17_reid_clean/bounding_box_train')
TEST_DIR = Path('dataset/mot17_reid_clean/bounding_box_test')

# -----------------------------------------------------------
# 1. Load model
# -----------------------------------------------------------
print('=' * 60)
print('MOT17 微调模型 - 完整评估（消除 batch-size 偏差）')
print('=' * 60)

model = create_model(num_classes=223, use_cbam=True, use_dilation=True)
ckpt = torch.load(CKPT, map_location='cpu')
if 'model_state_dict' in ckpt:
    model.load_state_dict(ckpt['model_state_dict'])
else:
    model.load_state_dict(ckpt)  # raw state_dict
model = model.to(DEVICE)
model.eval()
print(f'[OK] Loaded: {CKPT}')
print(f'     Keys: {"model_state_dict" if "model_state_dict" in ckpt else "raw state_dict"}')
print(f'     Total params: {sum(p.numel() for p in model.parameters()):,}')

# -----------------------------------------------------------
# 2. Evaluate on FULL TRAIN set (223 classes, no batch bias)
# -----------------------------------------------------------
transform = get_test_transforms()
train_ds = ReIDDataset(str(TRAIN_DIR), transform=transform, is_train=True)
train_loader = DataLoader(train_ds, batch_size=128, shuffle=False, num_workers=0)

tp = torch.zeros(223, dtype=torch.long)
fp = torch.zeros(223, dtype=torch.long)
fn = torch.zeros(223, dtype=torch.long)

print(f'\n[Evaluating FULL training set: {len(train_ds)} images, 223 classes]')

with torch.no_grad():
    for batch in tqdm(train_loader, desc='Train Eval'):
        imgs, labels = batch[0], batch[1]
        imgs = imgs.to(DEVICE)
        labels = labels.to(DEVICE)
        logits, _ = model(imgs)
        preds = logits.argmax(dim=1)

        for c in range(223):
            pred_c = (preds == c)
            true_c = (labels == c)
            tp[c] += (pred_c & true_c).sum().cpu()
            fp[c] += (pred_c & ~true_c).sum().cpu()
            fn[c] += (~pred_c & true_c).sum().cpu()

eps = 1e-8
per_class_p = tp.float() / (tp + fp).float().clamp(min=eps)
per_class_r = tp.float() / (tp + fn).float().clamp(min=eps)
per_class_f1 = 2 * per_class_p * per_class_r / (per_class_p + per_class_r + eps)

# Only count classes with samples
active = (tp + fn) > 0
n_active = active.sum().item()

# Macro (equally weight each class)
macro_p = per_class_p[active].mean().item()
macro_r = per_class_r[active].mean().item()
macro_f1 = 2 * macro_p * macro_r / (macro_p + macro_r + eps) if (macro_p + macro_r) > 0 else 0.0

# Micro (each sample equally weighted)
micro_tp = tp.sum().float()
micro_fp = fp.sum().float()
micro_fn = fn.sum().float()
micro_p = (micro_tp / (micro_tp + micro_fp + eps)).item()
micro_r = (micro_tp / (micro_tp + micro_fn + eps)).item()
micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r + eps) if (micro_p + micro_r) > 0 else 0.0

# Overall accuracy
total_correct = tp.sum().item()
total_samples = len(train_ds)
accuracy = total_correct / total_samples

print(f'\n--- 训练集（223 类）完整评估 ---')
print(f'  样本数:     {total_samples}')
print(f'  有效类别:   {n_active}/223')
print(f'')
print(f'  [Micro]  每个样本等权')
print(f'    Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)')
print(f'    Precision: {micro_p:.4f}')
print(f'    Recall:    {micro_r:.4f}')
print(f'    F1-Score:  {micro_f1:.4f}')
print(f'')
print(f'  [Macro]  每个类别等权 ({n_active} 类)')
print(f'    Precision: {macro_p:.4f}')
print(f'    Recall:    {macro_r:.4f}')
print(f'    F1-Score:  {macro_f1:.4f}')

# Sensitivity: how many classes have P/R > 0.5
good_p = (per_class_p[active] > 0.5).sum().item()
good_r = (per_class_r[active] > 0.5).sum().item()
print(f'  P>0.5 的类: {good_p}/{n_active} ({100*good_p/n_active:.1f}%)')
print(f'  R>0.5 的类: {good_r}/{n_active} ({100*good_r/n_active:.1f}%)')

# -----------------------------------------------------------
# 3. Evaluate on VAL set (56 unseen IDs — feature retrieval only)
# -----------------------------------------------------------
test_ds = ReIDDataset(str(TEST_DIR), transform=transform, is_train=False)
# Note: no pid_to_idx mapping — keep original val PIDs for retrieval eval
test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=0)

all_feats = []
all_labels = []
with torch.no_grad():
    for batch in tqdm(test_loader, desc='Val Feature Extract'):
        imgs, labels = batch[0], batch[1]
        imgs = imgs.to(DEVICE)
        _, feats = model(imgs)
        all_feats.append(F.normalize(feats, p=2, dim=1).cpu())
        all_labels.append(labels)
all_feats = torch.cat(all_feats, dim=0)   # (992, 2048)
all_labels = torch.cat(all_labels, dim=0)  # (992,)

# Retrieval metrics
dist_mat = torch.cdist(all_feats, all_feats, p=2)  # (992, 992)
# Exclude self-matches
dist_mat.fill_diagonal_(float('inf'))

sorted_idx = dist_mat.argsort(dim=1)
n = len(all_labels)

for k in [1, 5, 10, 20]:
    correct = 0
    for i in range(n):
        top_k = sorted_idx[i, :k]
        if all_labels[i] in all_labels[top_k]:
            correct += 1
    print(f'  Val Rank-{k}: {correct/n:.4f} ({100*correct/n:.2f}%)')

# mAP
aps = []
for i in range(n):
    q_label = all_labels[i]
    rel = (all_labels[sorted_idx[i]] == q_label).float()
    if rel.sum() == 0:
        continue
    precisions = torch.cumsum(rel, dim=0) / torch.arange(1, n, dtype=torch.float)
    ap = (precisions * rel).sum() / rel.sum()
    aps.append(ap.item())
val_map = np.mean(aps)
print(f'  Val mAP:   {val_map:.4f}')

# -----------------------------------------------------------
# 4. Summary
# -----------------------------------------------------------
print(f'\n{"="*60}')
print(f'  总结')
print(f'{"="*60}')
print(f'  训练集 (223 ID):')
print(f'    Micro  F1 = {micro_f1:.4f}  (<-- 每个样本等权，不受 batch 影响)')
print(f'    Macro  F1 = {macro_f1:.4f}  (<-- 每个 ID 等权)')
print(f'    Accuracy  = {accuracy:.4f}')
print(f'')
print(f'  验证集 (56 ID, 跨域检索):')
print(f'    Rank-1 = {correct/n:.4f}')
print(f'    mAP    = {val_map:.4f}')
print(f'{"="*60}')
