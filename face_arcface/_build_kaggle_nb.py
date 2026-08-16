"""生成 kaggle_train.ipynb: 内联 iresnet/arcface_loss/kaggle_convert/kaggle_train 四个脚本"""
import json
import pathlib

base = pathlib.Path(r'c:\D\Myproject\Data-processing\Hotel_Model_Train\face_arcface')

iresnet = (base / 'iresnet.py').read_text(encoding='utf-8')
arcface = (base / 'arcface_loss.py').read_text(encoding='utf-8')
convert = (base / 'kaggle_convert.py').read_text(encoding='utf-8')
train = (base / 'kaggle_train.py').read_text(encoding='utf-8')

cells = []

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# ArcFace 人脸识别训练 (Kaggle 版)\n",
        "\n",
        "IResNet50 + ArcFace, 训练 CASIA-WebFace。\n",
        "\n",
        "**数据是原始 `.rec`**(不是图片), 需先在本 notebook 里转换。用到的公开数据集:\n",
        "- 训练 `.rec`: `/kaggle/input/datasets/debarghamitraroy/casia-webface/casia-webface/train.rec` (+ train.idx)\n",
        "- 验证 `.bin`: `/kaggle/input/datasets/debarghamitraroy/casia-webface/eval`\n",
        "\n",
        "**步骤**: 依次运行代码块 → ① 转换 .rec → ② 冒烟训练确认 → ③ 正式训练。\n",
        "\n",
        "**注意**:\n",
        "- 需开启 **GPU 加速器** (右侧 Settings → Accelerator → GPU T4 x2)。\n",
        "- 转换约 20-40 分钟 (49 万张); 转完可「Create Dataset」存下图片, 下次免转换。\n",
        "- T4 16GB 显存, batch=256 可用; OOM 则改 128。\n",
        "- Kaggle 单次 GPU 会话约 **12h 上限**, 20 epoch 可能压线。\n",
        "- 训练产物在 `/kaggle/working/output/`, 会话结束不保留, 记得下载 `last.pt`。"
    ],
})

cells.append({
    "cell_type": "code",
    "metadata": {},
    "execution_count": None,
    "outputs": [],
    "source": ["%%writefile iresnet.py\n"] + iresnet.splitlines(keepends=True),
})

cells.append({
    "cell_type": "code",
    "metadata": {},
    "execution_count": None,
    "outputs": [],
    "source": ["%%writefile arcface_loss.py\n"] + arcface.splitlines(keepends=True),
})

cells.append({
    "cell_type": "code",
    "metadata": {},
    "execution_count": None,
    "outputs": [],
    "source": ["%%writefile kaggle_convert.py\n"] + convert.splitlines(keepends=True),
})

cells.append({
    "cell_type": "code",
    "metadata": {},
    "execution_count": None,
    "outputs": [],
    "source": ["%%writefile kaggle_train.py\n"] + train.splitlines(keepends=True),
})

cells.append({
    "cell_type": "code",
    "metadata": {},
    "execution_count": None,
    "outputs": [],
    "source": [
        "# ① 转换 .rec → ImageFolder (全量 ~49 万张, 约 20-40 分钟)\n",
        "!python kaggle_convert.py\n",
        "\n",
        "# 冒烟: 只转前 2000 张, 快速验证路径/格式 (通过后再跑上面全量)\n",
        "# !python kaggle_convert.py --limit 2000"
    ],
})

cells.append({
    "cell_type": "code",
    "metadata": {},
    "execution_count": None,
    "outputs": [],
    "source": [
        "# ② 冒烟训练: 1 epoch 只跑 3 个 batch, 确认身份数≈10572、loss 能下降\n",
        "!python kaggle_train.py --epochs 1 --batch 64 --max-steps 3\n",
        "\n",
        "# ③ 正式训练 (20 epoch): 冒烟通过后注释掉上面、取消下面这行\n",
        "# !python kaggle_train.py --epochs 20 --batch 256"
    ],
})

notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.10",
        },
    },
    "cells": cells,
}

out = base / 'kaggle_train.ipynb'
out.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding='utf-8')
print(f'[done] wrote {out}')
