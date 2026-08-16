"""
============================================================
人脸 Embedding 推理接口 (IResNet50 + ArcFace, 512 维)
用于「YOLO 框人 → yolov8n-face 框脸 → face_embed 认住客」跨天身份比对

用法:
  from face_arcface.face_embed import FaceEmbedder
  embedder = FaceEmbedder('face_arcface/output/last.pt')
  emb = embedder.embed('face.jpg')            # np.ndarray[512], L2 归一化
  sim = embedder.similarity(emb_a, emb_b)     # 余弦相似度 ∈ [-1,1]

命令行自测:
  python face_arcface/face_embed.py --model face_arcface/output/last.pt --img a.jpg --img2 b.jpg
============================================================
"""
import os
import sys

import numpy as np
import torch

# 脚本所在目录 = face_arcface/, 项目根取其父目录 (可移植, 不依赖 Windows 绝对路径)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(_SCRIPT_DIR)


class FaceEmbedder:
    """加载 ArcFace backbone, 输出 512 维 L2 归一化人脸 embedding"""

    MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    STD = np.array([0.5, 0.5, 0.5], dtype=np.float32)

    def __init__(self, checkpoint, device=None):
        sys.path.insert(0, _SCRIPT_DIR)
        from iresnet import iresnet50

        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
        # 兼容两种格式: 完整 checkpoint({'backbone':...}) 或 导出后的纯 backbone state_dict
        sd = ckpt['backbone'] if isinstance(ckpt, dict) and 'backbone' in ckpt else ckpt
        self.model = iresnet50(num_features=512).to(self.device)
        self.model.load_state_dict(sd)
        self.model.eval()
        self.checkpoint = checkpoint

    def _to_tensor(self, img_rgb):
        """RGB ndarray[H,W,3] → [1,3,112,112] tensor"""
        import cv2
        t = torch
        img = cv2.resize(img_rgb, (112, 112)).astype(np.float32) / 255.0
        x = t.from_numpy(img).permute(2, 0, 1)
        mean = t.from_numpy(self.MEAN).view(3, 1, 1)
        std = t.from_numpy(self.STD).view(3, 1, 1)
        x = (x - mean) / std
        return x.unsqueeze(0).to(self.device)

    def _load_image(self, img):
        """img 支持: 路径(str) / PIL.Image / numpy ndarray(BGR 或 RGB)"""
        import cv2
        if isinstance(img, str):
            img = cv2.imread(img, cv2.IMREAD_COLOR)   # BGR
        elif hasattr(img, 'convert'):                  # PIL
            img = np.array(img.convert('RGB'))
        else:                                          # ndarray
            img = np.asarray(img)
        if img is None:
            raise ValueError('无法读取图片')
        if img.ndim == 3 and img.shape[2] == 3:
            # 默认按 BGR 处理(cv2 读入), 传入 PIL/RGB 时自行 convert
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    @torch.no_grad()
    def embed(self, img, flip=False):
        """输入人脸图 → 512 维 L2 归一化 embedding (np.float32)"""
        rgb = self._load_image(img)
        x = self._to_tensor(rgb)
        f = self.model(x)
        if flip:
            f = f + self.model(torch.flip(x, dims=[3]))
            f = f / 2.0
        f = torch.nn.functional.normalize(f, p=2, dim=1)
        return f[0].cpu().numpy().astype(np.float32)

    @staticmethod
    def similarity(emb_a, emb_b):
        """余弦相似度 ∈ [-1, 1]"""
        a = np.asarray(emb_a, dtype=np.float32)
        b = np.asarray(emb_b, dtype=np.float32)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def main():
    def arg(name, default=''):
        try:
            return sys.argv[sys.argv.index(name) + 1]
        except (ValueError, IndexError):
            return default

    model = arg('--model', 'face_arcface/output/last.pt')
    img_a = arg('--img')
    img_b = arg('--img2', img_a)
    if not img_a:
        print('用法: python face_arcface/face_embed.py --model <ckpt> --img a.jpg [--img2 b.jpg]')
        sys.exit(1)

    embedder = FaceEmbedder(model)
    ea = embedder.embed(img_a)
    print(f'  embedding dim={ea.shape}  norm={np.linalg.norm(ea):.4f}  (已 L2 归一化)')
    if img_b != img_a:
        eb = embedder.embed(img_b)
        print(f'  余弦相似度: {embedder.similarity(ea, eb):.4f}')


if __name__ == '__main__':
    main()
