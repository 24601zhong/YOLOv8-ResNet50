"""
============================================================
导出 backbone-only 人脸 embedding 模型
去掉 ArcFace 分类头 + 优化器状态, 只留 IResNet50 backbone (512 维)

输入: face_arcface/output/last.pt         (392MB 完整 checkpoint)
输出: face_arcface/output/face_embed_iresnet50.pth   (仅 backbone state_dict, ~170MB)
可选: --onnx  同时导出 face_embed_iresnet50.onnx

用法:
  python face_arcface/export_face_embed.py --model face_arcface/output/last.pt
  python face_arcface/export_face_embed.py --model face_arcface/output/last.pt --onnx
============================================================
"""
import os
import sys

PROJECT = r'c:\D\Myproject\Data-processing\Hotel_Model_Train'


def arg(name, default=''):
    try:
        i = sys.argv.index(name)
        return sys.argv[i + 1]
    except (ValueError, IndexError):
        return default


def main():
    os.chdir(PROJECT)
    sys.path.insert(0, os.path.join(PROJECT, 'face_arcface'))
    import torch
    from iresnet import iresnet50

    MODEL = arg('--model', 'face_arcface/output/last.pt')
    OUT = arg('--out', 'face_arcface/output/face_embed_iresnet50.pth')

    print('=' * 64)
    print(f'  Face Embedding Export')
    print(f'  输入: {MODEL}')
    print('=' * 64)

    ckpt = torch.load(MODEL, map_location='cpu', weights_only=False)
    epoch = ckpt.get('epoch', '?')
    num_classes = ckpt.get('num_classes', '?')
    print(f'  完整 checkpoint: epoch={epoch}  num_classes={num_classes}')
    print(f'  keys: {list(ckpt.keys())}')

    backbone_sd = ckpt['backbone']
    n_params = sum(p.numel() for p in backbone_sd.values())
    print(f'  backbone 参数量: {n_params/1e6:.2f} M')

    torch.save(backbone_sd, OUT)
    size_mb = os.path.getsize(OUT) / 1e6
    print(f'  [导出] {OUT}  ({size_mb:.1f} MB)')

    # 校验: 用导出的 state_dict 重建模型, 前向输出与完整 checkpoint 一致
    m_export = iresnet50(num_features=512).eval()
    m_export.load_state_dict(backbone_sd)
    m_full = iresnet50(num_features=512).eval()
    m_full.load_state_dict(ckpt['backbone'])
    with torch.no_grad():
        dummy = torch.randn(2, 3, 112, 112)
        diff = (m_export(dummy) - m_full(dummy)).abs().max().item()
    print(f'  [校验] 导出模型 vs 完整模型 输出最大误差: {diff:.2e} (应≈0)')

    # 可选 ONNX
    if '--onnx' in sys.argv:
        onnx_path = OUT.replace('.pth', '.onnx')
        torch.onnx.export(
            m_export, torch.randn(1, 3, 112, 112), onnx_path,
            input_names=['face'], output_names=['embedding'],
            opset_version=14, dynamic_axes=None,
        )
        print(f'  [导出] {onnx_path}  ({os.path.getsize(onnx_path)/1e6:.1f} MB)')

    print('=' * 64)
    print('  Export Complete.')
    print('=' * 64)


if __name__ == '__main__':
    main()
