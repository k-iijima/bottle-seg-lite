#!/usr/bin/env bash
# マスクリファイン(refine_masks.py)用の軽量セットアップ。torch は base image のものを使う。
set -e
command -v git >/dev/null 2>&1 || (apt-get update && apt-get install -y --no-install-recommends git)
pip install -q "numpy<2" opencv-python-headless pycocotools pillow huggingface_hub einops psutil
[ -d /opt/sam3 ] || git clone --depth 1 https://github.com/facebookresearch/sam3.git /opt/sam3
pip install -q -e /opt/sam3
echo "[setup_refine] done"
