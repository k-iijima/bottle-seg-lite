# bottle データセット

ペットボトルのリアルタイム検出・インスタンスセグメンテーション（+属性認識）学習用データセット。
`datasets/bottle/` に配置。**現在進行中の作業状況は [DATASET_STATUS.md](DATASET_STATUS.md) を参照**。

## 1. 概要

| 項目 | 値 |
|---|---|
| 画像数 | **21,612**（全画像 `images/all/` にフラット配置） |
| インスタンス数 | bottle 120,244 / cap 29,253 / label 32,822（2026-07-11 機械修正 + SAM3 リファイン後） |
| クラス | 3クラス: `bottle`(1) / `cap`(2) / `label`(3) |
| 属性 | bottle ごとに10種（→ §5） |
| フォーマット | COCO instance segmentation（polygon 主体、iscrowd=1 の ignore 領域 333件 = RLE 284 + 散在ポリゴン 49） |
| 代表解像度 | 1280×720 (38%) / 640×480 (16%) ほか混在 |

## 2. ソースと来歴

| ソース | 画像数 | 備考 |
|---|---|---|
| COCO | 8,820 | bottle カテゴリを含む画像 |
| LVIS | 3,103 | bottle 系カテゴリ（COCO と画像重複は dedup 済み） |
| TACO | 263 | ごみ・散乱ボトル。同一画像263重複を統合済み |
| YouTube (CC) | 9,426 | Creative Commons 動画 **494本** からフレーム抽出 |

- ファイル名は `images/all/<source>_<id>.jpg`（例: `youtube_-RBS3mnXFOo_240.jpg`）。
- 由来の生 bbox は `instances_*_coco/_lvis/_taco.json` に provenance として保持。
- **dedup / リーク解消済み**（2026-06-27〜28）: TACO 重複統合、train↔test 跨ぎ90件のリーク解消
  （[dedup_merge.py](dedup_merge.py)）。
- YouTube 収集: CC 絞り込み検索 → DL → SAM3 "bottle" 判定 → CLIP 多様性 dedup → pod 間グローバル dedup
  （13,571→9,426枚採用、[collect_youtube.py](collect_youtube.py) / [merge_youtube_fleet.py](merge_youtube_fleet.py)）。

## 3. split 構成

| split | images | bottle | cap | label |
|---|---|---|---|---|
| train | 17,610 | 101,936 | 25,089 | 28,741 |
| val | 1,869 | 8,484 | 1,848 | 1,968 |
| test | 2,133 | 9,824 | 2,316 | 2,113 |
| **all** | **21,612** | **120,244** | **29,253** | **32,822** |

- YouTube 由来は**動画単位**で split 割当て（近接フレームの train/val/test リーク防止）。
- `instances_all_*.json` は全体、`instances_{train,val,test}_*.json` が学習用 split。

## 4. アノテーションファイル（ファイル接尾辞）

| 接尾辞 | 内容 | 生成方法 | 網羅範囲 |
|---|---|---|---|
| **`_sam3full`** | **正式アノテ: 3クラスマスク（bottle/cap/label）+ bottle 属性10種** | SAM3 seg（box→mask / テキスト部位分離 part-min=64）+ Qwen3-VL-30B-A3B VQA を [merge_parts_attrs.py](merge_parts_attrs.py) で統合 | 全21,612枚（属性は >=96px の44,332個体） |
| （接尾辞なし）/`_coco/_lvis/_taco` | ソース生 bbox | 各データセット由来 | provenance |

> 中間ブランチ `_sam3merge`（bottle 1クラス）/ `_sam3parts`（3クラス、属性なし）/
> `_sam3attr`（bottle+属性のみ）は 2026-07-11 に `_sam3full` へ統合・検証のうえ**削除済み**。
> 同日 [inspect_sam3full.py](inspect_sam3full.py) で品質検査し、[fix_sam3full.py](fix_sam3full.py) で
> 親リンク再割当・重複 NMS・浮遊 parts 削除等の機械修正を適用（詳細は DATASET_STATUS §4）。

**カスタムフィールド**（COCO 標準外）:

| フィールド | 付く場所 | 意味 |
|---|---|---|
| `seg_source` | annotation | 検出の出自: `sam3` / `sam3_text` / `sam3_part`(cap・label) / なし(ソース由来) |
| `seg_refined` | annotation | **マスクを SAM3 box→mask で再生成済み**（2026-07-11、>=24px 非crowd の 155,241件。segmentation/bbox/area が更新されている） |
| `parent_bottle_id` | cap/label | 部位の親 bottle の annotation id |
| `score` | cap/label | SAM3 検出スコア（0.4 以上のみ採用） |
| `attributes` | bottle | §5 の属性 dict |
| `attr_conf` | bottle | 属性の信頼度（CLIP backend 使用時の material のみ） |

## 5. 属性スキーマ（10種）

Qwen3-VL の VQA で bottle crop ごとに付与。各属性に `unknown` あり。
定義は [attribute_pipeline.py](attribute_pipeline.py) の `VLM_QUESTIONS`、CVAT 用は [cvat_labels_parts.json](cvat_labels_parts.json)。

| 属性 | 値 |
|---|---|
| `material` | pet / glass / can / other |
| `cap` | capped / uncapped |
| `cap_color` | none / white / black / red / blue / green / yellow / orange / silver / transparent |
| `label` | labeled / unlabeled |
| `label_color` | none / white / black / red / blue / green / yellow / orange / brown / multicolor |
| `fill_level` | empty / low / half / high / full |
| `crushed` | intact / crushed |
| `visibility` | full / occluded |
| `orientation` | upright / lying |
| `depiction` | **real / depicted**（実物か、絵・イラスト・印刷・画面上の描写か） |

- 付与対象は bbox 長辺 **>=96px** の bottle（属性値ありは 44,305 個体）。小さい個体は unknown のまま。
- 生成モデル: Qwen3-VL-30B-A3B-Instruct bf16（2026-07-11 実行）。`depiction=depicted` は 2,829件（6.4%）。
- VLM と SAM3 parts が矛盾した cap/label の presence・color（cap 8,380 / label 6,852 個体分）は
  2026-07-11 の機械修正で **unknown 化済み**（誤教師信号の除去。fix_sam3full.py §5）。
- 実行系: [attribute_pipeline.py](attribute_pipeline.py)（シャード分散対応）→ [merge_attrs.py](merge_attrs.py) で統合。
  RunPod フリートは [runpod/_attr_fleet.py](runpod/_attr_fleet.py)。

## 6. bottle サイズ分布（bbox 長辺, all）

| 閾値 | 個体数 |
|---|---|
| >=32px | 101,781 |
| >=48px | 84,367 |
| >=64px | 67,695 |
| >=96px | 44,334 |
| >=128px | 29,657 |
| >=256px | 8,977 |

## 7. ライセンス / 出典上の注意

- **COCO/LVIS**: 画像は Flickr 由来（各画像のライセンスに従う）。アノテーションは CC-BY 4.0。
- **TACO**: アノテーション CC-BY 4.0、画像は投稿者ライセンス混在。
- **YouTube**: Creative Commons (CC-BY) 表示の動画のみ収集。動画IDはファイル名に保持
  （`youtube_<videoID>_<frame>.jpg`）。
- 再配布・公開時は上記の帰属表示が必要。

## 8. 出力・再生成

```bash
# CVAT 用エクスポート（3クラス + 属性）
docker compose --profile tools run --rm seg \
  python export_coco_for_cvat.py --splits test --suffix _sam3full
```

- CVAT ラベル定義: [cvat_labels_parts.json](cvat_labels_parts.json)
  （※属性定義が古い: depiction / cap有無 / label有無 が未定義 → 検品前に要更新）
- 個別手順: [README_SAM3_local.md](README_SAM3_local.md) / [README_CVAT.md](README_CVAT.md) /
  [runpod/README_RUNPOD.md](runpod/README_RUNPOD.md)
