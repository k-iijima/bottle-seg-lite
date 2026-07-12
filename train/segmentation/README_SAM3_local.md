# SAM 3 でセグメンテーションを付与する（ローカル / RTX 4060）

既存の COCO bbox データセット `datasets/bottle` の **低品質な segmentation だけ** を、
SAM 3 の **box プロンプト** で生成し直して COCO instance segmentation 形式にし直します。

- スクリプト: [make_sam3_segmentation.py](make_sam3_segmentation.py)
- 出力: 元ファイルは触らず `instances_*_sam3seg.json` を新規作成（非破壊）

## なぜ box プロンプトか

既存アノテーションには **信頼できる bbox** が既にあります。テキスト（"pet bottle"）で
検出し直すと取りこぼし・重複・別物体の混入リスクがありますが、各 bbox を SAM 3 の
`predict_inst(box=...)` に渡せば **1 物体 = 1 マスク** で確実です。SAM 3 の SAM1 互換
インタラクティブ予測器（`enable_inst_interactivity=True`）を使います。

## 対象（既定）

| 種別 | 件数(all) | 既定 | 理由 |
|---|---|---|---|
| 矩形状の簡易ポリゴン（≤5点, iscrowd=0） | 492 | **再生成** | 実質 bbox の矩形。SAM3 で本物の輪郭にできる |
| 空の segmentation | 0 | 再生成 | （現状は無し） |
| RLE（このデータでは全て **iscrowd=1** の crowd 領域） | 284 | **スキップ** | crowd は個別インスタンスではなく、box→mask が無意味。RTMDet-Ins 等の学習でも iscrowd=1 は無視される。再生成すると壊れる |
| 良質なポリゴン（COCO/LVIS/TACO） | 約35,400 | 温存 | そのまま使う |

> crowd も再生成したい場合は `make_sam3_segmentation.py` の `REGEN_RLE = True` / `SKIP_ISCROWD = False` に変更。
> 全アノテーションを一律 SAM3 で作り直したい場合は、`needs_regen()` を `return True` に変えるだけで全件処理に切り替わります。

処理は `instances_all.json` を一度だけ実行し、annotation id 一致で
`instances_train/val/test.json` に自動伝播します（3 ファイルの ann id は all の部分集合であることを確認済み）。

## 実行（Docker / GPU）— 推奨

このプロジェクトでは Docker でビルド済みです（GPU 必須）。SAM 3 要件
（Python 3.12+ / PyTorch 2.7+ / CUDA 12.6+）はすべてイメージ内で解決します。
ホスト側に必要なのは **NVIDIA ドライバ + Docker（nvidia ランタイム有効）** のみ。

- イメージ定義: [docker/Dockerfile](docker/Dockerfile)（torch cu128 + SAM3 + 変換ツール）
- compose サービス `seg`: リポジトリ root の `docker-compose.yml`（`tools` プロファイル）
- HF 認証: root の `.env` の `HF_TOKEN` を compose が自動で渡し、スクリプトが login する
  （ゲート付き checkpoint は初回 run 時に DL → `hf_cache` ボリュームに永続化）

```bash
# リポジトリ root で
make seg-build     # イメージをビルド（初回のみ。torch cu128 で ~13GB）

make seg-preview   # 動作確認: 30画像だけ処理（= docker compose --profile tools run --rm seg）
make seg           # 全件処理（約492件 / 432画像、RTX 4060 で ~2-3分）
```

> RTX 4060 (8GB) で box→mask 推論は約 3.2 img/s。実測 GPU メモリは余裕あり。
> `triton: Failed to find C compiler` の警告は無害（任意の後処理をスキップするだけで結果に影響なし）。

## マスクの目視確認（プレビュー）

低品質対象は背景の小さなボトル（中央値 ~20×20px）が多く、全体プレビューでは見えにくいため、
**bbox 周辺をズームしたクロップ**を別途出力します（GPU 不要）。

```bash
# コンテナ内で（日本語パスのため Windows ローカルの cv2.imread は不可。コンテナ推奨）
docker compose --profile tools run --rm seg python preview_crops.py --n 50
```

- 出力: `datasets/bottle/qa_sam3/crops/crop_*.jpg` と一覧 `crops/_montage.jpg`
  （緑=SAM3 マスク, オレンジ=元の bbox）
- 全体プレビュー（再生成マスクを元画像上に重畳）は `qa_sam3/preview_*.jpg`

## （参考）Docker を使わずローカルで動かす場合

ホストに直接入れる場合の手順（非推奨・日本語パスで cv2 IO に注意）:

```bash
pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128
git clone https://github.com/facebookresearch/sam3.git && pip install -e ./sam3
pip install "numpy<2" opencv-python pycocotools tqdm pillow huggingface_hub einops psutil
hf auth login                          # または root .env の HF_TOKEN を利用
python make_sam3_segmentation.py --limit 30
```

主なオプション / 設定（スクリプト冒頭）:

- `--limit N` : 対象画像を N 枚に制限（動作確認）
- `--seg polygon|rle` : 出力形式。既定 `polygon`（既存の良質ポリゴンに合わせる）
- `--no-preview` : QA プレビューを作らない
- `MIN_BOX_IOU` : SAM3 マスクの外接矩形が入力 bbox と乖離したら失敗扱い→元を温存
- `BATCH_BOXES` : VRAM が厳しければ下げる（OOM 時は 1 に）

## 出力

```
datasets/bottle/annotations/
  instances_all_sam3seg.json
  instances_train_sam3seg.json
  instances_val_sam3seg.json
  instances_test_sam3seg.json
datasets/bottle/qa_sam3/
  preview_*.jpg              # 再生成マスクの目視確認
```

## （オプション）テキスト検出で未ラベルのボトルを追加（B案）

元データ(COCO/LVIS)は写っているボトルを全部はアノテーションしていない。`_sam3seg` を入力に、
SAM3 のテキスト検出(`"bottle"`)で **既存 box と重複しない新規ボトルだけ追加**する。

- スクリプト: [add_text_detections.py](add_text_detections.py)
- 既定: prompt=`bottle` / score>=0.4 / 既存との重複 IoU>=0.5 はスキップ / 新規同士 NMS IoU 0.7
- 新規 ann には `seg_source="sam3_text"` と `score` が付く。既存 ann は一切変更しない。
- ⚠️ `"pet bottle"` は SAM3 で検出0件になるため使わないこと（`"bottle"` か `"plastic bottle"`）。

```bash
docker compose --profile tools run --rm seg python add_text_detections.py --limit 12   # 動作確認
docker compose --profile tools run --rm seg python add_text_detections.py              # 全12,449画像(~2.5-3h)
```

出力: `instances_*_sam3merge.json`（= 既存温存 + 新規）, QA: `qa_sam3/merge/`（橙=既存, 緑=新規）。
実績: 36,270 → **58,568 anns**（+22,298, score中央値0.57）。RTMDet-Ins では `_sam3merge` を ann_file に指定。

再生成したアノテーションには `"seg_source": "sam3"` が付きます（温存分には付きません）。

## RTMDet-Ins / MMDetection から使う

学習用の正式な設定は [mmdet_configs/rtmdet-ins_s_pet_bottle.py](mmdet_configs/rtmdet-ins_s_pet_bottle.py)
（3クラス、`instances_*_trainready.json` を使用）。RunPod での実行手順は
[runpod/README_RUNPOD.md](runpod/README_RUNPOD.md) §0b を参照。

## 注意

- SAM 3 出力は擬似ラベルです。`qa_sam3/preview_*.jpg` で必ず目視確認し、明らかな
  失敗（`fallback` ログ件数）があれば `MIN_BOX_IOU` 等を調整、または該当画像のみ手修正してください。
- 元の `instances_*.json` は変更しません。やり直しは出力ファイルを消すだけで OK です。
