"""
============================================================
CASIA-WebFace (.rec) → ImageFolder 图片目录 转换脚本 (Kaggle 版)
纯 Python 解析 MXNet RecordIO, 不依赖 mxnet

.rec 二进制格式 (已验证):
  .idx  文本: 每行 "key\tbyte_offset"
  .rec  每个 record: magic(4B=0xced7230a) + lencode(4B) + header(24B) + JPEG data
        header = flag(int32) + label(int32) + id(int32) + id2(float32) + 8B padding
        label 在 header[4:8] (float32 存储, 转 int), <0 为 junk(对齐失败)

输出: dataset/face_casia/images/{label}/{key}.jpg  (ImageFolder 结构)

用法:
  python kaggle_convert.py                 # 全量 (~49 万张, 约 20-40 分钟)
  python kaggle_convert.py --limit 2000    # 冒烟: 只转前 N 个 record
============================================================
"""
import os
import struct
import sys
import time

# Kaggle 公开数据集默认路径
DEFAULT_REC = '/kaggle/input/datasets/debarghamitraroy/casia-webface/casia-webface/train.rec'
DEFAULT_IDX = '/kaggle/input/datasets/debarghamitraroy/casia-webface/casia-webface/train.idx'
DEFAULT_OUT = '/kaggle/working/dataset/face_casia/images'

MAGIC = 0xced7230a
HEADER_LEN = 24  # flag(int32) + label(int32) + id(int32) + id2(float32) + 8B padding


def arg(name, default):
    try:
        i = sys.argv.index(name)
        return sys.argv[i + 1]
    except (ValueError, IndexError):
        return default


def main():
    REC = arg('--rec', DEFAULT_REC)
    IDX = arg('--idx', DEFAULT_IDX)
    OUT = arg('--out', DEFAULT_OUT)
    limit = int(arg('--limit', 0))

    # 读 .idx 文本 → [(key, offset)]
    offsets = []
    with open(IDX, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            k, v = line.split('\t')
            offsets.append((int(k), int(v)))
    print(f'[idx] {len(offsets)} 个 record')

    if limit:
        offsets = offsets[:limit]
        print(f'[limit] 只处理前 {limit} 个 record')

    os.makedirs(OUT, exist_ok=True)

    n_written = 0
    n_junk = 0
    n_bad = 0
    labels = set()          # 既计数唯一 label, 也缓存已建目录 (避免每张图都 makedirs)
    t0 = time.time()

    f = open(REC, 'rb')
    for i, (key, off) in enumerate(offsets):
        f.seek(off)
        magic = struct.unpack('<I', f.read(4))[0]
        lencode = struct.unpack('<I', f.read(4))[0]
        if magic != MAGIC:
            n_bad += 1
            continue
        header = f.read(HEADER_LEN)
        # label 实际以 float32 存储 (0x3F800000 = 1.0), 用 '<f' 读再转 int
        label_f = struct.unpack('<f', header[4:8])[0]
        data_len = lencode - HEADER_LEN
        data = f.read(data_len)

        if label_f != label_f or label_f < 0:  # NaN 或负数 = junk
            n_junk += 1
            continue
        label = int(label_f)

        if data[:2] != b'\xff\xd8':  # 非 JPEG (尾部有 ~7047 个 8 字节浮点 junk record)
            n_bad += 1
            continue

        label_dir = os.path.join(OUT, str(label))
        if label not in labels:
            os.makedirs(label_dir, exist_ok=True)
            labels.add(label)
        with open(os.path.join(label_dir, f'{key}.jpg'), 'wb') as imgf:
            imgf.write(data)
        n_written += 1

        if (i + 1) % 50000 == 0:
            el = time.time() - t0
            print(f'  ... {i+1}/{len(offsets)} 处理中, {n_written} 张已写, {el:.1f}s')

    f.close()
    el = time.time() - t0
    print(f'\n[完成] 处理 {len(offsets)} 个 record, 耗时 {el:.1f}s')
    print(f'  写入图片: {n_written}')
    print(f'  junk(对齐失败,已跳过): {n_junk}')
    print(f'  坏 record(magic 不符/非JPEG): {n_bad}')
    print(f'  身份数(唯一 label): {len(labels)}')
    print(f'  输出目录: {OUT}')


if __name__ == '__main__':
    main()
