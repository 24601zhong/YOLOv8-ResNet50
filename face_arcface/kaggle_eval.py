"""
============================================================
ArcFace 人脸识别 1:1 验证集评估 (Kaggle 版, 可移植)
读取 insightface 官方 eval/*.bin (pickle: [jpeg_bytes列表, issame布尔列表])
  - lfw.bin      (LFW, 6000 pairs)   ← 验收标准 ≥ 99.5%
  - cfp_fp.bin   (CFP-Frontal-Profile)
  - cplfw.bin    (CPLFW)
  - agedb_30.bin (AgeDB-30)
  - 其余: calfw / cfp_ff / sllfw / talfw (参考)

iresnet.py 与本脚本同目录。

用法:
  python kaggle_eval.py --model output/last.pt --flip
============================================================
"""
import os
import sys
import pickle
import time
from io import BytesIO

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# Kaggle 公开数据集默认 eval 目录
DEFAULT_EVAL_DIR = '/kaggle/input/datasets/debarghamitraroy/casia-webface/eval'


def arg(name, default):
    try:
        i = sys.argv.index(name)
        return sys.argv[i + 1]
    except (ValueError, IndexError):
        return default


def main():
    MODEL = arg('--model', os.path.join(SCRIPT_DIR, 'output/last.pt'))
    EVAL_DIR = arg('--data', DEFAULT_EVAL_DIR)
    BATCH = int(arg('--batch', 128))
    FLIP = '--flip' in sys.argv
    BINS = arg('--bins', '').split(',') if arg('--bins', '') else \
        ['lfw.bin', 'cfp_fp.bin', 'cplfw.bin', 'agedb_30.bin',
         'calfw.bin', 'cfp_ff.bin', 'sllfw.bin', 'talfw.bin']

    import numpy as np
    import cv2
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from torchvision import transforms

    from iresnet import iresnet50

    if not torch.cuda.is_available():
        print('[ERROR] CUDA not available'); sys.exit(1)
    device = torch.device('cuda')
    print('=' * 64)
    print(f'  Face Verification Eval  model={MODEL}')
    print(f'  flip={FLIP}  device={torch.cuda.get_device_name(0)}')
    print('=' * 64)

    # ---------------- 模型 ----------------
    ckpt = torch.load(MODEL, map_location='cpu', weights_only=False)
    num_classes = ckpt.get('num_classes', '?')
    epoch = ckpt.get('epoch', '?')
    backbone = iresnet50(num_features=512).to(device)
    backbone.load_state_dict(ckpt['backbone'])
    backbone.eval()
    print(f'  [model] epoch={epoch}  num_classes={num_classes}  (仅用 backbone)')

    # ---------------- 预处理 (与训练一致, 无随机) ----------------
    MEAN = [0.5, 0.5, 0.5]
    STD = [0.5, 0.5, 0.5]

    def decode_preprocess(jpeg_bytes):
        """JPEG bytes → [3,112,112] float tensor (RGB, 归一化)"""
        arr = np.frombuffer(jpeg_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)          # BGR
        if img is None:
            img = np.zeros((112, 112, 3), np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (112, 112))
        t = torch.from_numpy(img).permute(2, 0, 1).float().div_(255.0)
        for c in range(3):
            t[c] = (t[c] - MEAN[c]) / STD[c]
        return t

    @torch.no_grad()
    def extract_features(images_tensor):
        """images_tensor: [N,3,112,112] → [N,512] L2 归一化"""
        feats = []
        n = images_tensor.size(0)
        for i in range(0, n, BATCH):
            x = images_tensor[i:i + BATCH].to(device)
            f = backbone(x)
            if FLIP:
                f = f + backbone(torch.flip(x, dims=[3]))
                f = f / 2.0
            f = torch.nn.functional.normalize(f, p=2, dim=1)
            feats.append(f.cpu())
        return torch.cat(feats, dim=0)

    def calc_accuracy(sims, issame):
        """余弦相似度 sims[配对], issame[bool] → 最优阈值下的 accuracy"""
        sims = sims.numpy().astype(np.float64)
        issame = np.asarray(issame, dtype=bool)
        best_acc, best_thresh = 0.0, 0.0
        for thresh in np.arange(-1.0, 1.0, 0.005):
            pred = sims > thresh
            acc = (pred == issame).mean()
            if acc > best_acc:
                best_acc, best_thresh = acc, thresh
        return best_acc, best_thresh

    # ---------------- 逐数据集评估 ----------------
    print(f'\n  {"数据集":>10s} | {"pairs":>6s} | {"acc":>7s} | {"阈值":>7s} | {"耗时":>7s}')
    print('  ' + '-' * 58)
    for bin_name in BINS:
        bin_path = os.path.join(EVAL_DIR, bin_name)
        if not os.path.exists(bin_path):
            print(f'  {bin_name:>10s} |  (跳过, 文件不存在)')
            continue
        t0 = time.time()
        with open(bin_path, 'rb') as f:
            bins, issame = pickle.load(f, encoding='bytes')
        n_pairs = len(issame)
        # 解码全部图片
        tensors = [decode_preprocess(b) for b in bins]
        imgs = torch.stack(tensors)                       # [2*n_pairs, 3,112,112]
        feats = extract_features(imgs)                    # [2*n_pairs, 512]
        feats = feats.view(n_pairs, 2, -1)
        sims = torch.nn.functional.cosine_similarity(feats[:, 0], feats[:, 1], dim=1)
        acc, thresh = calc_accuracy(sims, issame)
        el = time.time() - t0
        mark = '  ★ LFW' if bin_name == 'lfw.bin' else ''
        print(f'  {bin_name[:-4]:>10s} | {n_pairs:>6d} | {acc*100:>6.2f}% | {thresh:>7.3f} | {el:>6.1f}s{mark}')

    print('=' * 64)
    print('  Eval Complete.  (LFW 验收标准 ≥ 99.5%)')
    print('=' * 64)


if __name__ == '__main__':
    main()
