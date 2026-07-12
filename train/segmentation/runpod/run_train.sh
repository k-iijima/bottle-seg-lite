#!/usr/bin/env bash
# Pod 上で RTMDet-Ins-s の学習 → test 評価 → 成果物 tar 化まで行う。
#   GPUS=8 bash run_train.sh                      # 8GPU DDP（lr は実績値 2.5e-4 を自動設定）
#   GPUS=8 bash run_train.sh --smoke              # 1 epoch だけ回して配線確認
#   GPUS=8 bash run_train.sh visualizer.vis_backends.1.run_name=h100x8_60e
# --smoke 以外の引数は --cfg-options の key=val として渡る。
set -e
. /workspace/venv/bin/activate
cd /workspace/mmdetection
export PYTHONPATH=/workspace/mmdetection:$PYTHONPATH
export MPLBACKEND=Agg
# RunPod コンテナでの NCCL 対策（8xH100 で実測済み）:
#   - NVLS(NVLink SHARP)は 3GPU 以上で "operation cannot be performed in the present
#     state" で死ぬ（コンテナに multicast 権限がない）→ 無効化が必須
#   - P2P 有効だと 8rank で不安定（6/8 しか成功しない）→ 無効化して SHM 経由に
export NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-0}
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
# MLflow 3.x は file ストアがオプトイン制（未設定だと MlflowException で落ちる）
export MLFLOW_ALLOW_FILE_STORE=true

CFG=/workspace/mmdet_configs/rtmdet-ins_s_pet_bottle.py
WORK=/workspace/work_pet_bottle
GPUS=${GPUS:-1}

mkdir -p /workspace/checkpoints
[ -f /workspace/checkpoints/rtmdet-ins_s_coco.pth ] || \
  wget -q -O /workspace/checkpoints/rtmdet-ins_s_coco.pth \
  https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet-ins_s_8xb32-300e_coco/rtmdet-ins_s_8xb32-300e_coco_20221121_212604-fdc5d7ec.pth

CFGOPTS=()
for a in "$@"; do
  if [ "$a" = "--smoke" ]; then
    CFGOPTS+=(train_cfg.max_epochs=1 train_cfg.val_interval=1)
  else
    CFGOPTS+=("$a")
  fi
done
# lr 未指定時のみ自動設定。COCO 事前学習からのファインチューンでは from-scratch 公式値
# (0.004@256) はもちろん、その 1/4 の 0.001@8GPU でも train loss が正常なまま val が崩壊
# （SyncBN running 統計が追従不能。TRAINING_LOG.md §4 参照）
# → 実績値 2.5e-4@8GPU（公式値の 1/16）を線形スケールで既定にする
if ! printf '%s\n' "${CFGOPTS[@]}" | grep -q 'optimizer\.lr='; then
  CFGOPTS+=(optim_wrapper.optimizer.lr=$(python -c "print(0.00025 * 32 * $GPUS / 256)"))
fi
ARGS=(--amp --work-dir "$WORK")
[ ${#CFGOPTS[@]} -gt 0 ] && ARGS+=(--cfg-options "${CFGOPTS[@]}")

if [ "$GPUS" -gt 1 ]; then
  bash tools/dist_train.sh "$CFG" "$GPUS" "${ARGS[@]}"
else
  python tools/train.py "$CFG" "${ARGS[@]}"
fi

BEST=$(ls -t "$WORK"/best_coco_segm_mAP_epoch_*.pth 2>/dev/null | head -1)
[ -z "$BEST" ] && BEST=$(ls -t "$WORK"/epoch_*.pth | head -1)
echo "[test] using $BEST"
python tools/test.py "$CFG" "$BEST" --work-dir "$WORK/test" 2>&1 | tail -30

cd /workspace
FILES=(mmdet_configs/rtmdet-ins_s_pet_bottle.py "${BEST#/workspace/}")
for f in work_pet_bottle/*.log work_pet_bottle/*/vis_data/scalars.json \
         work_pet_bottle/test/*.json; do
  [ -e "$f" ] && FILES+=("$f")
done
[ -d mlruns ] && FILES+=(mlruns)
tar czf train_outputs.tar.gz "${FILES[@]}"
echo "[run_train] done -> /workspace/train_outputs.tar.gz"
