#!/usr/bin/env bash
# RTMDet-Ins 学習環境のセットアップ（RunPod PyTorch イメージ上）。
# Colab で動作検証済みの組合せ（RTMDet-Ins検証.ipynb）を venv に再現する:
#   torch 2.1.0 cu121 / numpy 1.26.4 / mmcv 2.1.0 / mmdetection v3.3.0
set -e

python -m venv /workspace/venv
. /workspace/venv/bin/activate
pip install -q -U pip "setuptools<70" wheel

pip install -q torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
pip install -q --force-reinstall --no-cache-dir "numpy==1.26.4"
pip install -q --no-deps "opencv-python-headless==4.8.1.78"
pip install -q addict yapf termcolor rich pyyaml matplotlib packaging pillow \
               pycocotools scipy shapely six terminaltables tqdm mlflow

pip install -q mmengine==0.10.4
# ⚠️ openmmlab 配布の mmcv wheel（cu121/torch2.1 の 2.1.0/2.2.0 とも）は sm_90 カーネルを
# 含まず H100 で "no kernel image" になる → ソースビルド必須（ninja+32並列で ~10分）
pip install -q ninja
TORCH_CUDA_ARCH_LIST="9.0" MMCV_WITH_OPS=1 FORCE_CUDA=1 MAX_JOBS=32 \
  pip install -q mmcv==2.1.0 --no-binary mmcv --no-build-isolation
# mlflow 等が numpy 2.x を引っ張るため、最後に必ず 1.26.4 へ固定し直す
pip install -q --force-reinstall --no-deps "numpy==1.26.4"

command -v git >/dev/null 2>&1 || (apt-get update && apt-get install -y --no-install-recommends git)
[ -d /workspace/mmdetection ] || git clone -b v3.3.0 --depth 1 https://github.com/open-mmlab/mmdetection.git /workspace/mmdetection
pip install -q --no-build-isolation -e /workspace/mmdetection

python - <<'PY'
import numpy, torch, mmcv, mmengine, mmdet
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      "| numpy", numpy.__version__, "| mmcv", mmcv.__version__,
      "| mmdet", mmdet.__version__)
PY
echo "[setup_train] done"
