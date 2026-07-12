#!/usr/bin/env bash
# 属性分類器学習のセットアップ(RunPod PyTorch 2.4 イメージ上)。
# torch/torchvision はイメージ同梱のものをそのまま使う(mmcv 不要なので venv も不要)。
set -e
pip install -q pillow tqdm onnx onnxruntime timm
python - <<'PY'
import torch, torchvision, PIL
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      "gpus", torch.cuda.device_count(), "| tv", torchvision.__version__)
PY
echo "[setup_attrcls] done"
