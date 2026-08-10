import sys
import os

# 切换到项目根目录
os.chdir(r'c:\D\Myproject\Data-processing\Hotel_Model_Train\resnet50_reid_train')

# 重定向输出到日志文件，避免 tqdm 刷屏导致进程被杀
log_file = open('train_output/reid_log/training.log', 'a', buffering=1)
sys.stdout = log_file
sys.stderr = log_file

from train_reid import train_market1501
from pathlib import Path

print(f"\n{'='*60}")
print(f"ResNet50 ReID Training — Market-1501")
print(f"Start time: {__import__('datetime').datetime.now()}")
print(f"{'='*60}\n")

# Market-1501 数据集路径
market_dir = r'c:\D\Myproject\Data-processing\Market-1501-v15.09.15'

# 从之前训练的最佳模型续训（若存在）
pretrained = None
best_path = Path('train_output/reid_log/best_market1501_improved.pth')
if best_path.exists():
    pretrained = str(best_path)
    print(f"[续训] 加载已有最佳模型: {pretrained}")

train_market1501(
    market1501_dir=market_dir,
    output_dir='train_output/reid_log',
    device='cuda',
    num_epochs=120,
    batch_size=64,
    lr=3.5e-4,
    pretrained_weights=pretrained,
    use_cbam=True,
    use_dilation=True,
)

print(f"\n{'='*60}")
print(f"Training completed at: {__import__('datetime').datetime.now()}")
print(f"{'='*60}\n")

log_file.close()
