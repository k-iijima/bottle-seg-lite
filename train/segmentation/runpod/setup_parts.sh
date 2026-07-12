#!/usr/bin/env bash
# 部位分離(segment_parts.py)だけに必要な軽量セットアップ。torch は base image の 2.4.1 を使う
# （SAM3 推論は 2.4.1 で動作確認済み）。属性用の重い依存は入れない＝速い。
set -e
command -v git >/dev/null 2>&1 || (apt-get update && apt-get install -y --no-install-recommends git)
pip install -q "numpy<2" opencv-python-headless pycocotools pillow tqdm huggingface_hub einops psutil
[ -d /opt/sam3 ] || git clone --depth 1 https://github.com/facebookresearch/sam3.git /opt/sam3
pip install -q -e /opt/sam3
echo "[setup_parts] done"
