#!/usr/bin/env bash
# RunPod の PyTorch テンプレ（torch+CUDA 済み）上で属性パイプラインの依存を入れる。
set -e
command -v git >/dev/null 2>&1 || (apt-get update && apt-get install -y --no-install-recommends git)
pip install -U pip
# torch>=2.7 (cu128) を保証（RunPod の古い base image 対策）
python - <<'PY'
import subprocess, sys
try:
    import torch
    from packaging.version import parse as V
    need = V(torch.__version__.split('+')[0]) < V('2.7')
except Exception:
    need = True
if need:
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'torch==2.10.0', 'torchvision',
                    '--index-url', 'https://download.pytorch.org/whl/cu128'], check=True)
PY
pip install "transformers>=4.57" accelerate "qwen-vl-utils[decord]" bitsandbytes open_clip_torch
pip install "numpy<2" opencv-python-headless pycocotools pillow tqdm huggingface_hub einops psutil

# SAM 3（部位分離 segment_parts.py 用）。cwd に置くと import が衝突するので /opt に置く。
if [ ! -d /opt/sam3 ]; then
  git clone --depth 1 https://github.com/facebookresearch/sam3.git /opt/sam3
fi
pip install -e /opt/sam3

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
echo "[setup] done"
