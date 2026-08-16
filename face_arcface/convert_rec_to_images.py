"""
============================================================
CASIA-WebFace (.rec) → ImageFolder 图片目录 转换脚本
纯 Python 解析 MXNet RecordIO，不依赖 mxnet（避免 numpy 2.0 冲突）

已验证的 .rec 二进制格式:
  .idx  文本: 每行 "key\tbyte_offset"
  .rec  每个 record:
        magic(4B, 0xced7230a) + lencode(4B) + header(24B) + JPEG data(lencode-24 B)
        header = flag(int32) + label(int32) + id(int32) + id2(float32) + 8B padding
        label 在 header[4:8] = 身份 ID (0~10571),  <0 为 junk(对齐失败)

输出: dataset/face_casia/images/{label}/{key}.jpg  (ImageFolder 结构)

用法:
  python convert_rec_to_images.py            # 全量
  python convert_rec_to_images.py --limit 200   # 冒烟: 只转前 N 个 record
============================================================
"""
import os
import struct
import sys
import time
from pathlib import Path

PROJECT = r'c:\D\Myproject\Data-processing\Hotel_Model_Train'
REC = os.path.join(PROJECT, 'dataset/archive/casia-webface/train.rec')
IDX = os.path.join(PROJECT, 'dataset/archive/casia-webface/train.idx')
OUT = os.path.join(PROJECT, 'dataset/face_casia/images')

MAGIC = 0xced7230a
HEADER_LEN = 24  # flag(int32) + label(int32) + id(int32) + id2(float32) + 8B padding


def main():
    os.chdir(PROJECT)
    limit = None
    if '--limit' in sys.argv:
        limit = int(sys.argv[sys.argv.index('--limit') + 1])

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
    labels = set()
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
        os.makedirs(label_dir, exist_ok=True)
        with open(os.path.join(label_dir, f'{key}.jpg'), 'wb') as imgf:
            imgf.write(data)
        labels.add(label)
        n_written += 1

        if (i + 1) % 50000 == 0:
            el = time.time() - t0
            print(f'  ... {i+1}/{len(offsets)} 处理中, {n_written} 张已写, {el:.1f}s')

    f.close()
    el = time.time() - t0
    print(f'\n[完成] 处理 {len(offsets)} 个 record, 耗时 {el:.1f}s')
    print(f'  写入图片: {n_written}')
    print(f'  junk(对齐失败,已跳过): {n_junk}')
    print(f'  坏 record(magic 不符): {n_bad}')
    print(f'  身份数(唯一 label): {len(labels)}')
    print(f'  输出目录: {OUT}')


if __name__ == '__main__':
    main()
