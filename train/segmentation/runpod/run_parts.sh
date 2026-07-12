#!/usr/bin/env bash
# Pod 上で bottle + cap + label の3クラス部位分離を実行し、出力 JSON を tar 化。
# 使い方: HF_TOKEN=hf_xxx bash runpod/run_parts.sh [追加引数は segment_parts.py へ]
set -e
: "${HF_TOKEN:?HF_TOKEN を環境変数で渡してください (export HF_TOKEN=hf_...)}"

python segment_parts.py --data-root pet_bottle "$@"

cd pet_bottle/annotations
tar czf /workspace/parts_outputs.tar.gz \
  instances_all_sam3parts.json \
  instances_train_sam3parts.json \
  instances_val_sam3parts.json \
  instances_test_sam3parts.json
echo "[run_parts] done -> /workspace/parts_outputs.tar.gz  (runpodctl send で持ち帰り)"
