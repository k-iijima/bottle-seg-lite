#!/usr/bin/env bash
# 学習用の入力（画像 + trainready JSON + config + スクリプト）を tar にまとめる。
# seg コンテナ内で実行:
#   docker compose --profile tools run --rm seg bash runpod/package_train_inputs.sh
set -e
cd /work
mkdir -p runpod
tar cf runpod/train_inputs.tar \
  -C /work/datasets \
    bottle/images/all \
    bottle/annotations/instances_train_trainready.json \
    bottle/annotations/instances_val_trainready.json \
    bottle/annotations/instances_test_trainready.json \
  -C /work \
    mmdet_configs/rtmdet-ins_s_pet_bottle.py \
    runpod/setup_train.sh runpod/run_train.sh
echo "[package] created runpod/train_inputs.tar"
du -h runpod/train_inputs.tar
