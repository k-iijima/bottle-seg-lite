#!/usr/bin/env bash
# Pod 上で属性付与を実行し、出力 JSON だけを tar にまとめる。
# 使い方: HF_TOKEN=hf_xxx bash runpod/run.sh  [追加引数は attribute_pipeline.py に渡る]
set -e
: "${HF_TOKEN:?HF_TOKEN を環境変数で渡してください (export HF_TOKEN=hf_...)}"

python attribute_pipeline.py --data-root bottle "$@"

cd bottle/annotations
tar czf /workspace/attr_outputs.tar.gz \
  instances_all_sam3attr.json \
  instances_train_sam3attr.json \
  instances_val_sam3attr.json \
  instances_test_sam3attr.json
echo "[run] done -> /workspace/attr_outputs.tar.gz  (runpodctl send で持ち帰り)"
