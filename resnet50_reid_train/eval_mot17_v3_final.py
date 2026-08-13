#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Independent final evaluation of the ReID V3 model on MOT17 val.

This is deliberately a SEPARATE implementation from the training script's
`evaluate_reid` (which uses Euclidean distance). Here we use cosine similarity
on L2-normalized features to cross-check the reported Rank-1/mAP, and we
re-load the checkpoint + data from scratch.
"""
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from model_v3 import create_model_v3
from train_reid import ReIDDataset

CKPT = SCRIPT_DIR / "train_output" / "mot17_v3" / "best_mot17_v3.pth"
MOT17_DIR = PROJECT_DIR / "dataset" / "mot17_reid_all"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    print("=" * 60)
    print("  Independent ReID V3 Final Evaluation (MOT17 val)")
    print("=" * 60)

    # ---- transforms: identical to training test transforms ----
    transform = T.Compose([
        T.Resize((384, 128)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # ---- dataset (independent pid mapping, cross-domain) ----
    dataset = ReIDDataset(
        str(MOT17_DIR), transform=transform, is_train=False, pid_to_idx=None
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=False,
                        num_workers=0, pin_memory=True)
    print(f"\n[Data] {len(dataset)} images, {dataset.num_classes} IDs")

    # ---- model ----
    model = create_model_v3(
        num_classes=dataset.num_classes, use_cbam=True, use_dilation=True,
        arc_scale=30.0, arc_margin=0.4,
    )
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    if "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
    elif "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    else:
        sd = ckpt
    # Drop ArcFace head (classifier) — not used for feature extraction
    sd = {k: v for k, v in sd.items() if not k.startswith("arcface.")}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[Model] loaded. missing={len(missing)} unexpected={len(unexpected)}")
    if "best_rank1" in ckpt:
        print(f"[Checkpoint] training-reported best Rank-1 = {ckpt['best_rank1']:.4f}")
    if "best_mAP" in ckpt:
        print(f"[Checkpoint] training-reported best mAP    = {ckpt['best_mAP']:.4f}")

    model = model.to(DEVICE)
    model.eval()

    # ---- extract L2-normalized features ----
    feats, labels = [], []
    with torch.no_grad():
        for imgs, lbl, _ in loader:
            imgs = imgs.to(DEVICE)
            f = model(imgs, return_feature=True)   # [B, 2048] L2-normalized
            feats.append(f.cpu())
            labels.append(lbl)
    feats = torch.cat(feats, dim=0)
    labels = torch.cat(labels, dim=0).numpy()
    N = feats.size(0)
    print(f"[Features] {N} x {feats.size(1)}")

    # ---- independent metric: cosine similarity (feats already L2-normalized) ----
    sim = torch.mm(feats, feats.t()).numpy()   # [N, N] cosine similarity
    np.fill_diagonal(sim, -np.inf)             # exclude self-match

    order = np.argsort(-sim, axis=1)           # most similar first; self (=-inf) sorts LAST
    order = order[:, :-1]                      # drop self (last column) -> matches train's indices[i][1:]

    K = 20
    cmc = np.zeros(K)
    aps = []
    for i in range(N):
        ranked_labels = labels[order[i]]
        pos_mask = (ranked_labels == labels[i])
        num_pos = int((labels == labels[i]).sum()) - 1  # exclude self
        if num_pos <= 0:
            aps.append(0.0)
            continue
        # CMC
        cum = pos_mask[:K].cumsum()
        for k in range(K):
            if cum[k] >= 1:
                cmc[k] += 1.0
        # Average Precision
        correct = 0.0
        prec = []
        for k, m in enumerate(pos_mask, 1):
            if m:
                correct += 1
                prec.append(correct / k)
        aps.append(float(np.mean(prec)) if prec else 0.0)

    cmc = cmc / N
    mAP = float(np.mean(aps))

    print("\n" + "=" * 60)
    print("  FINAL INDEPENDENT METRICS (MOT17 val)")
    print("=" * 60)
    print(f"  Images : {N}   IDs: {dataset.num_classes}")
    print(f"  Rank-1 : {cmc[0]:.4f}")
    print(f"  Rank-5 : {cmc[4]:.4f}")
    print(f"  Rank-10: {cmc[9]:.4f}")
    print(f"  Rank-20: {cmc[19]:.4f}")
    print(f"  mAP    : {mAP:.4f}")
    print("=" * 60)

    verdict = "PASS" if cmc[0] > 0.9 else "FAIL"
    print(f"\n  Target Rank-1 > 0.9 : {verdict}  (Rank-1 = {cmc[0]:.4f})")


if __name__ == "__main__":
    main()
