#!/usr/bin/env bash
# YouTube 収集 pod の軽量セットアップ（RunPod PyTorch base 上。torch は base のものを使用）。
set -e
command -v git >/dev/null 2>&1 || (apt-get update && apt-get install -y --no-install-recommends git)
command -v ffmpeg >/dev/null 2>&1 || (apt-get update && apt-get install -y --no-install-recommends ffmpeg)
pip install -q "numpy<2" opencv-python-headless pycocotools pillow tqdm huggingface_hub einops psutil \
    open_clip_torch yt-dlp
[ -d /opt/sam3 ] || git clone --depth 1 https://github.com/facebookresearch/sam3.git /opt/sam3
pip install -q -e /opt/sam3
echo "[setup_youtube] done"
