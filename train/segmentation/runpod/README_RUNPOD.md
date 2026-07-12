# RunPod でマスクリファイン / 属性自動付与を回す

## 0a. マスクリファイン（2026-07-11 仕様、H100 フリート）

`_sam3full` の全マスク（bottle/cap/label、長辺>=24px・非crowd 約15.6万）を
**既存 bbox を box プロンプトにした SAM3 再生成**で品質向上する。再検出はしないため
ann id / 属性 / parent_bottle_id / 件数は不変で、segmentation/bbox/area のみ更新。

- 実装: [../refine_masks.py](../refine_masks.py)（長辺<128px は拡大crop推論、bf16 autocast、
  安全ガード: 面積>=16 / bbox IoU>=0.3 / 旧マスク IoU>=0.2 を満たさなければ旧マスク温存）
- フリート: [_refine_fleet.py](_refine_fleet.py)（H100 優先、_attr_fleet と同じ運用）
- 反映: [../merge_refined.py](../merge_refined.py)（old_iou 分布レポート付き、
  変化の大きい上位500件を qa_sam3full/refine_low_iou.json に出力）

```bash
# 1) ローカル: 入力パッケージ（images + _sam3full → runpod/refine_inputs.tar ~4GB）
docker compose --profile tools run --rm seg bash runpod/package_refine_inputs.sh
# 2) フリート（8 pod x 2 worker = 16シャード）
python runpod/_refine_fleet.py create 8 2
python runpod/_refine_fleet.py seed
python runpod/_refine_fleet.py dispatch
python runpod/_refine_fleet.py poll          # 完走まで
python runpod/_refine_fleet.py heal          # 未完シャードがあれば
python runpod/_refine_fleet.py fetch         # -> runpod/refined/refined_*.json
python runpod/_refine_fleet.py term          # ★必ず削除（課金）
# 3) ローカル: 反映（dry-run で分布確認 → 適用）
python merge_refined.py --dry-run
python merge_refined.py
python inspect_sam3full.py                   # 再検査
```

ローカル動作確認: `docker compose --profile tools run --rm seg python refine_masks.py --limit 30 --qa 16`
（プレビュー: `qa_sam3full/refine_preview/`、赤=新マスク・緑=旧輪郭）

---

## 0b. RTMDet-Ins-s 学習（2026-07-11 準備完了、単一 H100）

`_sam3full` ベースの学習用アノテ（[../make_train_anns.py](../make_train_anns.py) で
`depicted` を iscrowd=1 化した `instances_*_trainready.json`）で
RTMDet-Ins-s を COCO 事前学習から 60 epoch ファインチューンする。

- 設定: [../mmdet_configs/rtmdet-ins_s_pet_bottle.py](../mmdet_configs/rtmdet-ins_s_pet_bottle.py)
  （3クラス、バッチ32、lr 5e-4、最後10epochで重い増強オフ、save_best=segm_mAP。
  mmengine でのパース検証済み）
- 環境: torch 2.1.0 cu121 / mmcv 2.1.0 / mmdetection v3.3.0（Colab 検証済みの組合せを venv 再現）
- **MLflow 記録**: `MLflowVisBackend`（file ストア、サーバ不要）で loss/mAP/config を
  `/workspace/mlruns` に自動記録 → train_outputs.tar.gz に同梱して持ち帰り。
  ローカル閲覧: 各 run の mlruns/ を `train/segmentation/mlruns/` に累積展開して
  `pip install mlflow && mlflow ui --backend-store-uri "file:train/segmentation/mlruns"`。
  run 名は `--cfg-options visualizer.vis_backends.1.run_name=<名前>` で指定推奨
- 実績（2026-07-11、8×H100 60epoch）: **test bbox_mAP 0.376 / segm_mAP 0.352**
  （対 SAM3 疑似ラベルの再現度。人手 GT ではない）。
  `GPUS=8 bash run_train.sh` で dist_train + NCCL 対策 + lr 自動設定まで面倒を見る。
- **⚠️ lr は 2.5e-4 固定（run_train.sh の既定）**。公式 from-scratch 値 0.004 や 0.001 では
  train loss が正常のまま val が崩壊する（SyncBN running 統計が高 lr に追従できない。実測済み）。
- 複数 pod での DDP は不可（pod 間ネットワークで勾配同期が破綻）。必ず 1 pod のマルチ GPU で。
- その他の環境罠は run_train.sh / setup_train.sh のコメント参照
  （NCCL NVLS 無効化 / mmcv sm_90 ソースビルド / MLFLOW_ALLOW_FILE_STORE / numpy 1.26 固定）。

```bash
# 1) ローカル: 学習アノテ生成 + 入力パッケージ（runpod/train_inputs.tar ~4GB）
python make_train_anns.py --data-root datasets/bottle
docker compose --profile tools run --rm seg bash runpod/package_train_inputs.sh
# 2) H100 pod を1台作成（RunPod PyTorch テンプレ、ports 22/tcp、disk 80GB）し、scp で転送
scp -i runpod/.rp/id_rsa -P <PORT> runpod/train_inputs.tar root@<IP>:/workspace/
ssh  -i runpod/.rp/id_rsa -p <PORT> root@<IP> "cd /workspace && tar xf train_inputs.tar && mv -f runpod/*.sh . && bash setup_train.sh"
# 3) スモーク（1 epoch で配線確認）→ 本走
ssh ... "cd /workspace && nohup bash run_train.sh --smoke > train.log 2>&1 &"
ssh ... "cd /workspace && nohup bash run_train.sh > train.log 2>&1 &"
# 4) 回収して pod 削除
scp -i runpod/.rp/id_rsa -P <PORT> root@<IP>:/workspace/train_outputs.tar.gz runpod/
```

> ⚠️ seed/転送は **ネイティブ scp を使う**（paramiko SFTP は速度劣化・二重起動事故の実績あり）。
> 学習後の ONNX 化（アプリ deploy 用）は mmdeploy で別途 — `make rtmdet-onnx`（model/rtmdet/）。

---

# （既存）属性自動付与（material + cap/label）

> ⚠️ **歴史的手順**（実行済み・通常は再実行不要）。当時は9属性・`_sam3attr` 出力だったが、
> 現在は10属性（[../DATASET.md](../DATASET.md) §5）で `_sam3full` に統合済み。
> 再実行する場合は suffix・属性数を現状に合わせて読み替えること。

ローカル(8GB)では VLM が遅いので、cap/label を含む属性付与は RunPod の大きい GPU で実行する。
**画像は戻さず、戻りは属性付き JSON だけ**（数MB）なので軽い。

- パイプライン本体: [../attribute_pipeline.py](../attribute_pipeline.py)（VRAM 自動判定。>=~18GB は bf16、未満は 4bit）
- material=CLIP ViT-H-14 / cap・label=Qwen3-VL-8B（`--vlm-model` で変更可）

## 0. 事前

- RunPod アカウントと API キー（リポジトリ root `.env` の `RUNPOD_KEY`）
- ローカルとPodにそれぞれCLIを入れる: https://github.com/runpod/runpodctl
  ```bash
  runpodctl config --apiKey "$RUNPOD_KEY"     # ローカル（pod 管理用。send/receive 自体は不要）
  ```

## 1. ローカル: 入力をパッケージ化

```bash
docker compose --profile tools run --rm seg bash runpod/package_inputs.sh
# -> train/segmentation/runpod/attr_inputs.tar.gz （画像~1–2GB込み）
```

## 2. RunPod: Pod を起動

- テンプレート: **RunPod PyTorch**（torch+CUDA 済み）
- モデル別の目安（`attribute_pipeline.py` が VRAM とモデルサイズを見て 4bit/バッチを自動調整）:

  | モデル | 精度 | GPU目安 | ディスク | DL |
  |---|---|---|---|---|
  | `Qwen3-VL-8B-Instruct`（既定・バランス） | 良 | 24GB(4090) | 40GB | ~16GB |
  | `Qwen3-VL-30B-A3B-Instruct`（**精度重視・推奨**, MoEで高速） | 高 | 48GB(A6000/L40S) or 24GB(4bit) | 120GB | ~60GB |
  | `Qwen3-VL-32B-Instruct`（dense・最高精度） | 最高 | 80GB or 48GB(4bit) | 120GB | ~64GB |

  > 大モデルは安全側で自動 4bit になる。MoE の 30B-A3B は稼働 3B 相当で **32B 級精度を高速**。

## 3. 転送（runpodctl send/receive）

ローカル:
```bash
runpodctl send train/segmentation/runpod/attr_inputs.tar.gz
# -> 表示される受信コード（例: 1234-foo-bar-baz）をコピー
```
Pod（Web ターミナル等）:
```bash
cd /workspace
runpodctl receive <受信コード>
tar xzf attr_inputs.tar.gz        # -> bottle/ と attribute_pipeline.py, runpod/
```

## 4. Pod: セットアップ＆実行

```bash
cd /workspace
bash runpod/setup.sh
export HF_TOKEN=hf_xxxxxxxx        # ゲート付き checkpoint 取得に必要

# 精度重視（推奨）: material も VLM が判定（CLIP より文脈に強い）
bash runpod/run.sh --vlm-model Qwen/Qwen3-VL-30B-A3B-Instruct --material-backend vlm --vlm-min 96

# バランス（既定 8B / material は CLIP, cap・label は VLM）
# bash runpod/run.sh --vlm-min 96
# まず少数で品質確認:
# bash runpod/run.sh --vlm-model Qwen/Qwen3-VL-30B-A3B-Instruct --material-backend vlm --vlm-min 96 --limit-vlm 300
# -> /workspace/attr_outputs.tar.gz （instances_*_sam3attr.json のみ）
```

## 4b. （別ジョブ）bottle + cap + label の3クラス部位分離

属性とは別に、SAM3 で **cap / label を部位として分離**し 3クラス
(bottle=1, cap=2, label=3) の COCO を作る。Pod 上で:

```bash
export HF_TOKEN=hf_xxxxxxxx
bash runpod/run_parts.sh --part-min 96      # >=96px の bottle に cap/label を付与
# まず試す: bash runpod/run_parts.sh --part-min 160 --limit 100
# -> /workspace/parts_outputs.tar.gz （instances_*_sam3parts.json）
```

- 各 bottle を crop → SAM3 テキスト検出（cap="bottle cap"/"lid", label="label"/"product label"）の
  top マスクを全体座標へ戻して別インスタンス化。`parent_bottle_id` と `seg_source="sam3_part"` 付き。
- CVAT ラベル定義は3クラス版 [../cvat_labels_parts.json](../cvat_labels_parts.json) を使う。
- 持ち帰り後の CVAT export は `--suffix _sam3parts`:
  ```bash
  docker compose --profile tools run --rm seg \
    python export_coco_for_cvat.py --splits test --suffix _sam3parts
  ```
- 属性(_sam3attr)と部位(_sam3parts)は別ブランチ。両方使う場合は別タスク/別プロジェクトで取り込むのが簡単。

## 5. 持ち帰り

Pod:
```bash
runpodctl send /workspace/attr_outputs.tar.gz   # 受信コードをコピー
```
ローカル:
```bash
cd train/segmentation/datasets/bottle/annotations
runpodctl receive <受信コード>
tar xzf attr_outputs.tar.gz        # instances_*_sam3attr.json が annotations/ に展開
```

## 6. CVAT へ（属性付き）

```bash
docker compose --profile tools run --rm seg \
  python export_coco_for_cvat.py --splits test --suffix _sam3attr --only-source sam3_text sam3
```
CVAT の Project ラベルは [../cvat_labels_parts.json](../cvat_labels_parts.json)（bottle/cap/label + 10属性定義済み。旧 cvat_labels.json は削除済み）を Raw に貼る。

## 属性は9種（`--attrs` で選択可）
material / cap / cap_color / label / label_color / fill_level / crushed / visibility / orientation
（各 unknown あり。VLM の生成回数 ≒ 対象数 × 属性数 なので、**時間は属性数に比例**）

```bash
# 軽くしたい時は属性を絞る
bash runpod/run.sh --vlm-model Qwen/Qwen3-VL-30B-A3B-Instruct --material-backend vlm \
     --attrs material,cap,label,fill_level --vlm-min 96
```

## メモ / 見積り
- 対象数: `>=64px` 約18.6k / `>=96px` 約9.9k / `>=128px` 約5.9k（× 属性数 = 生成回数）。
  小物は認識モデルに使えないので、まず `--vlm-min 96`〜`128` で始めるのが効率的。
- 9属性フルは generations が多い → 大GPU + 30B-A3B(MoE)が現実的。まず `--limit-vlm 300` で品質確認推奨。
- material を CLIP にすると（`--material-backend clip`）その分 VLM 質問が1つ減って速い。
- Pod は使い終わったら必ず停止（課金）。出力 JSON はローカルに取得済みなら Pod 破棄でOK。
