#!/usr/bin/env bash
# ローカルで入力（画像+merge JSON+スクリプト）を tar にまとめる。
# seg コンテナ内で実行する想定:
#   docker compose --profile tools run --rm seg bash runpod/package_inputs.sh
set -e
cd /work
mkdir -p runpod
# 注意: GNU tar の -C は「直前の -C からの相対」。絶対パスにして compound を防ぐ。
tar czf runpod/attr_inputs.tar.gz \
  -C /work/datasets \
    bottle/images/all \
    bottle/annotations/instances_all_sam3merge.json \
    bottle/annotations/instances_train_sam3merge.json \
    bottle/annotations/instances_val_sam3merge.json \
    bottle/annotations/instances_test_sam3merge.json \
  -C /work \
    attribute_pipeline.py segment_parts.py runpod/setup.sh runpod/run.sh runpod/run_parts.sh
echo "[package] created runpod/attr_inputs.tar.gz"
du -h runpod/attr_inputs.tar.gz
