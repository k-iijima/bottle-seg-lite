#!/usr/bin/env bash
# Pod 上で属性分類器を学習する。ARCHS のモデルを GPU に振り分けて並列実行。
#   bash run_attrcls.sh            # 本走(30 epoch)
#   bash run_attrcls.sh --smoke    # 1 epoch で配線確認
#   ARCHS="mobilenetv4_conv_small mobilenetv4_conv_medium" bash run_attrcls.sh
# 終了後 /workspace/attrcls_outputs.tar.gz に best.pth / metrics / ONNX をまとめる。
set -e
cd /workspace
EPOCHS=30
[ "$1" = "--smoke" ] && EPOCHS=1
ARCHS=${ARCHS:-"mobilenet_v3_small mobilenet_v3_large"}

NGPU=$(python -c "import torch;print(torch.cuda.device_count())")
i=0
for arch in $ARCHS; do
  CUDA_VISIBLE_DEVICES=$((i % NGPU)) python train_attr_cls.py train \
    --crops-root attr_crops --arch "$arch" --epochs "$EPOCHS" \
    > "train_$arch.log" 2>&1 &
  i=$((i + 1))
done
wait

rm -f attrcls_outputs.tar.gz
tar czf attrcls_outputs.tar.gz work_attr train_*.log
echo "[run_attrcls] done -> /workspace/attrcls_outputs.tar.gz"
