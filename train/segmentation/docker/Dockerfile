# SAM 3 (image, box-prompt) でセグメンテーションを生成するための GPU コンテナ。
# host driver 経由で GPU を使う想定（CUDA toolkit はイメージに含めず、torch 同梱の CUDA を利用）。
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/hf_cache \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# system deps: git(sam3 clone), opencv ランタイム
RUN apt-get update && apt-get install -y --no-install-recommends \
        git ca-certificates libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# PyTorch (CUDA 12.8 build) — SAM 3 要件: torch>=2.7
RUN pip install --upgrade pip \
    && pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128

# SAM 3 本体（推論に必要な base 依存のみ）
RUN git clone --depth 1 https://github.com/facebookresearch/sam3.git /opt/sam3 \
    && pip install -e /opt/sam3

# 変換 / QA まわり（numpy は SAM3 要件で <2 に固定）
RUN pip install "numpy<2" opencv-python-headless pycocotools tqdm pillow huggingface_hub einops psutil
# 属性の自動アノテ用（material=CLIP ゼロショット, cap/label=VQA VLM）
RUN pip install open_clip_torch
RUN pip install "transformers>=4.57" accelerate "qwen-vl-utils[decord]" bitsandbytes
# YouTube 収集（collect_youtube.py）用: 動画 DL と読み込み
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && pip install yt-dlp
# RunPod fleet オーケストレーション（SSH/SFTP）用
RUN pip install paramiko

WORKDIR /work
# 既定は動作確認（30画像・プレビュー生成）。全件は compose 側で command を上書き。
CMD ["python", "make_sam3_segmentation.py", "--limit", "30"]
