"""生成 kaggle_eval.ipynb: 内联 iresnet.py + kaggle_eval.py"""
import json
import pathlib

base = pathlib.Path(r'c:\D\Myproject\Data-processing\Hotel_Model_Train\face_arcface')

iresnet = (base / 'iresnet.py').read_text(encoding='utf-8')
evalpy = (base / 'kaggle_eval.py').read_text(encoding='utf-8')

cells = []

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# ArcFace 评估 (Kaggle 版)\n",
        "\n",
        "在 Kaggle 上评估 LFW / CFP-FP / CPLFW / AgeDB 等 1:1 验证集。\n",
        "\n",
        "**前置条件**: 需要一个训练好的 `last.pt`(含 `backbone` state_dict)。三种方式之一:\n",
        "1. **同一会话**: 先运行训练 notebook,`/kaggle/working/output/last.pt` 已生成, 直接跑本 notebook。\n",
        "2. **上传模型**: 把本地 `last.pt` 拖到右侧 Data → `Add file`, 它会出现在 `/kaggle/working/`。\n",
        "3. **存成数据集**: 训练完用「Create Dataset」把 `last.pt` 存成新数据集, 下次 `Add Input` 引用。\n",
        "\n",
        "**验证数据已内置**: `/kaggle/input/datasets/debarghamitraroy/casia-webface/eval`\n",
        "\n",
        "**注意**: 需开启 **GPU 加速器** (T4 即可)。"
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
    "source": ["%%writefile kaggle_eval.py\n"] + evalpy.splitlines(keepends=True),
})

cells.append({
    "cell_type": "code",
    "metadata": {},
    "execution_count": None,
    "outputs": [],
    "source": [
        "# 评估: 默认模型 /kaggle/working/output/last.pt, 加 --flip 镜像增强\n",
        "!python kaggle_eval.py --model output/last.pt --flip\n",
        "\n",
        "# 若模型在别处, 指定路径:\n",
        "# !python kaggle_eval.py --model /kaggle/working/last.pt --flip"
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

out = base / 'kaggle_eval.ipynb'
out.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding='utf-8')
print(f'[done] wrote {out}')
