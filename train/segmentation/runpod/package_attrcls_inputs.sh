#!/usr/bin/env bash
# 属性分類器学習の入力(クロップ+jsonl+スクリプト)を tar にまとめる。
# 画像全体ではなく抽出済みクロップ(~0.4GB)だけ送るので軽い。
# train/segmentation で実行: bash runpod/package_attrcls_inputs.sh
set -e
cd "$(dirname "$0")/.."
tar cf runpod/attrcls_inputs.tar \
  -C datasets/bottle attr_crops \
  -C "$PWD" train_attr_cls.py \
            runpod/setup_attrcls.sh runpod/run_attrcls.sh
echo "[package] created runpod/attrcls_inputs.tar"
du -h runpod/attrcls_inputs.tar
