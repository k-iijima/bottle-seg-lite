# CVAT で `_sam3full` を検品・補正する

正式アノテ `instances_*_sam3full.json`（bottle/cap/label 3クラス + bottle 10属性）を CVAT に
取り込み、目視で確認・修正して COCO 形式で書き戻すワークフロー。
機械検査・機械修正は適用済み（[DATASET_STATUS.md](DATASET_STATUS.md) §4）。CVAT で見るのは
機械判定できなかった残りが中心。

CVAT スタックは VSLAM の `docker-compose-cvat.yml`（CVAT v2.68.0）を流用しており、
**VSLAM と同一インスタンス**（同じ DB・アカウント）。
⚠️ ボリューム削除等のリセットは VSLAM のプロジェクトも消えるため行わないこと。

> 2026-07-11 以前に取り込んだ旧 pet_bottle プロジェクト/タスク（`_sam3parts` 版、属性なし）は
> 古いので **CVAT の UI 上で削除してよい**。以下の zip から新規に取り込み直す。

## 1. CVAT を起動

```bash
make cvat-up            # = docker compose -f docker-compose-cvat.yml up -d --build
# 初回のみ管理ユーザー作成（VSLAM で作成済みならスキップ）
make cvat-superuser
```

ブラウザで http://localhost:4810 を開きログイン。`make cvat-down` で停止。

## 2. Project 作成（ラベル定義）

1. CVAT で **Project** を作成。
2. ラベルは **Raw** エディタに [cvat_labels_parts.json](cvat_labels_parts.json) を貼付
   （bottle に10属性、cap / label は属性なし。属性は bottle 側にのみ持つ設計）。
   これが無いと import 時に属性がマッチしない。

## 3. 取り込む zip（生成済み）

`datasets/pet_bottle/cvat/` に出力済み。Project の **Actions → Import dataset →
形式 COCO 1.0** で zip を指定すると subset 名の Task ができる。

| zip | 画像数 | 用途 |
|---|---|---|
| `review_multicap_cvat_coco.zip` | 1,136 | **優先**: 同一 bottle に離れた cap が複数残る画像。ゴミ cap の削除・正しい cap の確認 |
| `review_attr_cvat_coco.zip` | 292 | 属性矛盾で unknown 化した bottle のサンプル（cap/label の presence・color を再判定） |
| `val_cvat_coco.zip` | 1,869 | val 全量の一般検品（マスク品質・属性） |
| `test_cvat_coco.zip` | 2,133 | test 全量の一般検品（同上） |

再生成・追加セットは [export_coco_for_cvat.py](export_coco_for_cvat.py)（既定 suffix `_sam3full`）:

```bash
python export_coco_for_cvat.py --splits val test              # split 全量
python export_coco_for_cvat.py --splits train                 # train 全量（~3GB、必要時のみ）
python export_coco_for_cvat.py --splits all --subset review_xxx \
    --image-list <basename一覧.txt>                            # 任意の検品セット
python export_coco_for_cvat.py --splits test --only-source sam3_part   # seg_source で絞る
```

検品リストの元データ: `datasets/pet_bottle/qa_sam3full/report.json`（機械修正前の疑義全件）/
`report_after_fix.json`（修正後に残る疑義）。リスト生成例は
`qa_sam3full/review_*_images.txt` を参照。

## 4. 検品の観点

- **review_multicap**: 1 本のボトルに cap は物理的に 1 個。余分な cap マスク（ゴミ・隣のボトル由来）
  を削除。積み重なったボトルでは正しい親のボトルを確認。
- **review_attr**: VLM と SAM3 が食い違ったため `cap`/`cap_color`/`label`/`label_color` を
  unknown 化してある。実物を見て正しい値を入れ直す。
- **val / test**: YouTube 由来の cap/label マスク品質、`fill_level`（unknown 58%）、
  `depiction`（real/depicted 境界例）を重点的に。iscrowd=1（ignore 領域）は無視推奨。

## 5. 修正して書き戻す

1. CVAT で確認・補正（不要マスク削除、輪郭修正、未検出ボトルの追加、属性入力）。
2. Project/Task の **Actions → Export dataset** → 形式 **COCO 1.0** → zip をダウンロード。
3. zip 内 `annotations/instances_<subset>.json` が補正済み COCO。
   `instances_<subset>_reviewed.json` として `annotations/` に置き、`_sam3full` へ反映する
   （`file_name` は basename なので、学習時の `data_prefix=images/all/` と整合）。

## メモ

- CVAT は `images/<subset>/<basename>` と `file_name`(basename) を照合して画像対応づけする。
  接頭辞や subset を file_name に残すと「画像に何も乗らない / No media data found」になる。
- 本データの basename は source 接頭辞付き（例 `coco_train2017_15239.jpg`）で全体一意。
  TACO 由来の同名重複は export 時に `__id<image_id>` 付与で自動一意化される。
- iscrowd=1（RLE 284 + 散在ポリゴン 49 = 333件）も取り込まれるが ignore 領域なので修正不要。
- `score` / `seg_source` / `parent_bottle_id` は COCO 標準外のトップレベル項目のため CVAT では
  見えない（往復で失われる点に注意。書き戻し時は元 JSON とのマージで温存すること）。
