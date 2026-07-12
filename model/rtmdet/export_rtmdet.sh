#!/usr/bin/env bash
# RTMDet-Ins-s（bottle データセット学習済み）を ONNX 化する（コンテナ内で実行）。
# 入力（マウント）:
#   /train  = train/segmentation（ckpt: work_pet_bottle/..., config: mmdet_configs/...）
#   /out    = app/assets/models（rtmdet_ins.onnx を出力）
# 検証: mmdeploy の deploy.py が torch と onnxruntime の出力可視化を /out/verify/ に保存する。
set -e

SIZE=${SIZE:-256}                 # モバイル向け入力解像度（正方）
SCORE_THR=${SCORE_THR:-0.35}
TOPK=${TOPK:-10}                  # モバイルでは masks 出力 [1,K,S,S] が支配的コスト → 小さく
CKPT=${CKPT:-/train/work_pet_bottle/best_coco_segm_mAP_epoch_60.pth}
MODEL_CFG=/train/mmdet_configs/rtmdet-ins_s_pet_bottle.py
DEMO_IMG=${DEMO_IMG:-/train/datasets/bottle/images/all/coco_train2017_10799.jpg}

# モデル側 test_cfg でも件数と閾値を絞る（rtmdet-ins はこちらが効く）
cat > /tmp/model_cfg.py <<EOF
_base_ = ['$MODEL_CFG']
model = dict(test_cfg=dict(score_thr=$SCORE_THR, max_per_img=$TOPK))
EOF
MODEL_CFG=/tmp/model_cfg.py

# デプロイ config: mmdeploy 同梱の rtmdet-ins 用 static config を継承して
# 入力解像度と後処理（件数・閾値）だけ上書きする
cat > /tmp/deploy_cfg.py <<EOF
_base_ = ['/opt/mmdeploy/configs/mmdet/instance-seg/instance-seg_rtmdet-ins_onnxruntime_static-640x640.py']
onnx_config = dict(input_shape=($SIZE, $SIZE))
codebase_config = dict(post_processing=dict(
    score_threshold=$SCORE_THR,
    max_output_boxes_per_class=$TOPK,
    pre_top_k=1000,
    keep_top_k=$TOPK,
))
EOF

mkdir -p /out/verify
python /opt/mmdeploy/tools/deploy.py \
  /tmp/deploy_cfg.py "$MODEL_CFG" "$CKPT" "$DEMO_IMG" \
  --work-dir /out/verify --device cuda --dump-info

cp /out/verify/end2end.onnx /out/rtmdet_ins.onnx
python - <<'PY'
import onnx, os
m = onnx.load('/out/rtmdet_ins.onnx')
print('[onnx] inputs :', [(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim]) for i in m.graph.input])
print('[onnx] outputs:', [(o.name, [d.dim_value for d in o.type.tensor_type.shape.dim]) for o in m.graph.output])
print('[onnx] size   :', os.path.getsize('/out/rtmdet_ins.onnx')//1_000_000, 'MB')
PY
echo "[export] done -> /out/rtmdet_ins.onnx（可視化: /out/verify/output_onnxruntime.jpg）"
