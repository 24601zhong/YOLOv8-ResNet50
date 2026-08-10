#!/bin/bash
# ============================================================
# 环境搭建脚本：创建 conda 虚拟环境并安装所有依赖
# 使用方式：bash setup_env.sh
# ============================================================

set -e

ENV_NAME="hotel_det_reid"
PYTHON_VERSION="3.9"

echo "=============================================="
echo "  酒店行人检测与重识别 - 环境搭建"
echo "  环境名称: ${ENV_NAME}"
echo "  Python 版本: ${PYTHON_VERSION}"
echo "=============================================="

# Step 1: 创建 conda 虚拟环境
echo ""
echo "[Step 1/3] 创建 conda 虚拟环境..."
conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y

# Step 2: 激活环境
echo ""
echo "[Step 2/3] 激活环境..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${ENV_NAME}

# Step 3: 安装 PyTorch (CUDA 11.8)
echo ""
echo "[Step 3/3] 安装 Python 依赖包..."
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Step 4: 安装其余依赖
pip install ultralytics==8.0.200
pip install opencv-python==4.8.1.78
pip install albumentations
pip install pillow
pip install numpy
pip install scipy
pip install imagehash
pip install labelImg
pip install matplotlib
pip install seaborn
pip install scikit-learn
pip install faiss-cpu

echo ""
echo "=============================================="
echo "  环境安装完成！"
echo "  请运行以下命令验证环境："
echo ""
echo "  conda activate ${ENV_NAME}"
echo "  python -c \"import torch, cv2, ultralytics; print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available())\""
echo "=============================================="
