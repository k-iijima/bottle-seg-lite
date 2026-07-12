#!/usr/bin/env bash
# 属性付与(attribute_pipeline.py, VLM backend)だけに必要なセットアップ。
# SAM3 / open_clip は入れない（material も VLM で判定する運用）。
set -e
pip install -q -U pip
# torch>=2.7 (cu128) を保証（RunPod の古い base image 対策; setup.sh と同じ手順）
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
# base image の torchaudio は旧 torch 向けで import が壊れる（transformers が拾って落ちる）ため除去
pip uninstall -y -q torchaudio 2>/dev/null || true
pip install -q "transformers>=4.57" accelerate "qwen-vl-utils[decord]" bitsandbytes
pip install -q "numpy<2" pillow tqdm "huggingface_hub[hf_transfer]" psutil
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
echo "[setup_attrs] done"
