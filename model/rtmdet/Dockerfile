# RTMDet-Ins-s (mmdetection v3.3.0) を mmdeploy で ONNX にエクスポートするコンテナ。
# ⚠️ mmdeploy の rtmdet-ins エクスポートは CUDA 前提（CPU だと 'cuda' 直書きで落ちる）
#    → GPU(host driver) + cu121 版 torch/mmcv を使う。学習時と同じ mm 系バージョン。
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN apt-get update && apt-get install -y --no-install-recommends \
        git ca-certificates libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip "setuptools<70" wheel \
 && pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121 \
 && pip install "numpy==1.26.4" opencv-python-headless==4.8.1.78 pycocotools shapely six terminaltables \
        addict yapf termcolor rich pyyaml matplotlib packaging pillow tqdm

RUN pip install mmengine==0.10.4 \
 && pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html

# 学習 config の _base_ が /workspace/mmdetection を参照するため、同じパスに clone する
RUN git clone -b v3.3.0 --depth 1 https://github.com/open-mmlab/mmdetection.git /workspace/mmdetection \
 && pip install --no-build-isolation -e /workspace/mmdetection

RUN git clone -b v1.3.1 --depth 1 https://github.com/open-mmlab/mmdeploy.git /opt/mmdeploy \
 && pip install mmdeploy==1.3.1 onnxruntime==1.16.3 onnx

# numpy を最後に再固定（依存が 2.x を引き込むのを防ぐ）
RUN pip install --force-reinstall --no-deps "numpy==1.26.4"

# fp16 変換（convert_fp16.py）用
RUN pip install onnxconverter-common

# onnxruntime 1.16 の .so は実行可能スタックを要求しており、新しめの
# Docker/WSL カーネルだと import 時に "cannot enable executable stack" で
# 落ちる → GNU_STACK フラグをクリアしておく
RUN python - <<'EOF' \
 && python -c "import onnxruntime; print('onnxruntime import OK')"
# ELF の GNU_STACK セグメントから実行フラグ (PF_X) を落とす
import glob, struct
path = glob.glob('/usr/local/lib/python3.10/site-packages/onnxruntime/capi/'
                 'onnxruntime_pybind11_state*.so')[0]
with open(path, 'r+b') as f:
    data = bytearray(f.read(64))
    assert data[:4] == b'\x7fELF' and data[4] == 2, 'ELF64 expected'
    e_phoff = struct.unpack_from('<Q', data, 0x20)[0]
    e_phentsize = struct.unpack_from('<H', data, 0x36)[0]
    e_phnum = struct.unpack_from('<H', data, 0x38)[0]
    for i in range(e_phnum):
        off = e_phoff + i * e_phentsize
        f.seek(off)
        ph = f.read(8)
        p_type, p_flags = struct.unpack('<II', ph)
        if p_type == 0x6474E551:  # PT_GNU_STACK
            f.seek(off + 4)
            f.write(struct.pack('<I', p_flags & ~0x1))
            print(f'cleared execstack: flags {p_flags:#x} -> {p_flags & ~0x1:#x}')
            break
    else:
        print('no PT_GNU_STACK segment found (nothing to do)')
EOF

WORKDIR /work
