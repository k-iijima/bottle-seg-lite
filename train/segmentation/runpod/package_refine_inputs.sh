#!/usr/bin/env bash
# マスクリファイン用の入力（画像 + _sam3full + スクリプト）を tar にまとめる。
# seg コンテナ内で実行:
#   docker compose --profile tools run --rm seg bash runpod/package_refine_inputs.sh
set -e
cd /work
mkdir -p runpod
tar cf runpod/refine_inputs.tar \
  -C /work/datasets \
    pet_bottle/images/all \
    pet_bottle/annotations/instances_all_sam3full.json \
  -C /work \
    refine_masks.py runpod/setup_refine.sh
echo "[package] created runpod/refine_inputs.tar"
du -h runpod/refine_inputs.tar
