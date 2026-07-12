# pet_bottle データセット 最新状態（2026-07-11 時点）

ペットボトル検出/セグメンテーション用データセット `datasets/pet_bottle` の現況まとめ。
**データセット自体の恒久的な仕様（構成・スキーマ・ライセンス）は [DATASET.md](DATASET.md)**、
個別手順は [README_SAM3_local.md](README_SAM3_local.md) / [README_CVAT.md](README_CVAT.md) /
[runpod/README_RUNPOD.md](runpod/README_RUNPOD.md) を参照。

## 1. 構成（マージ後）

| split | images | bottle | cap | label |
|---|---|---|---|---|
| all | 21,612 | 120,244 | 29,253 | 32,822 |
| train | 17,610 | 101,936 | 25,089 | 28,741 |
| val | 1,869 | 8,484 | 1,848 | 1,968 |
| test | 2,133 | 9,824 | 2,316 | 2,113 |

（2026-07-11 の機械修正 + SAM3 マスクリファイン適用後。→ §4）

**ソース内訳**
- coco / lvis / taco 由来: 12,186 枚（dedup 後）
- YouTube (Creative Commons) 由来: 9,426 枚（494 動画から抽出）

`images/all/` に全画像をフラット配置（`<source>_<id>.jpg`）。枚数=21,612 でアノテと一致。

## 2. アノテーションのブランチと網羅状況

| ファイル接尾辞 | 内容 | 生成方法 | 網羅範囲 |
|---|---|---|---|
| **`_sam3full`** | **正式アノテ（唯一）: 3クラスマスク + bottle 10属性** | [merge_parts_attrs.py](merge_parts_attrs.py) で `_sam3parts` + `_sam3attr` を統合（2026-07-11） | **全21,612枚** |
| `instances_*.json`（接尾辞なし）/ `_coco/_lvis/_taco` | ソース生 bbox | 各データセット由来 | provenance |

**今後の学習・CVAT補正は `_sam3full` を正とする**（bottle 120,313 / cap 29,535 / label 33,094、
属性付き bottle 44,305）。属性は bottle アノテーションのみに持ち、cap/label は
`parent_bottle_id` で親を参照する（cap_color 等の複製による食い違いを防ぐため）。

**2026-07-11 データ整理（削除済み）** — `_sam3full` への包含を全 split で検証のうえ削除:
- 中間ブランチ `_sam3merge` / `_sam3parts` / `_sam3attr`（×4 split）、`youtube_sam3merge`（統合済みステージング）
- `annotations/predupe_backup/` / `annotations/youtube_premerge_backup/`（6月のマージ前バックアップ）
- `cvat/*.zip`（6/28 出力の属性なし旧版。`_sam3full` から再エクスポートすること）
| `instances_*.json`（接尾辞なし）/ `_coco/_lvis/_taco` | ソース生 bbox | 各データセット由来 | provenance |

### 属性の10種
`material / cap(有無) / cap_color / label / label_color / fill_level / crushed / visibility /
orientation / depiction(実物か絵・印刷か)`（各 unknown あり）。
[attribute_pipeline.py](attribute_pipeline.py)（シャード分散対応）で Qwen3-VL-30B-A3B により生成、
[merge_attrs.py](merge_attrs.py) で統合。値の一覧は [DATASET.md](DATASET.md) §5。

**2026-07-11 に全量再付与完了**（旧 _sam3attr は実質スモークのみで5件しか値が無かった）:
- 対象: bbox 長辺 >=96px の bottle 44,334個体 → **44,332 付与**（2件は画像読込失敗、unknown のまま）
- RunPod フリート: RTX PRO 6000 (96GB) ×6 worker + 4090 hub、bf16/バッチ8 で **2.6〜2.9 crop/s/台**、
  推論 44〜50分/台（[runpod/_attr_fleet.py](runpod/_attr_fleet.py)）
- `depiction=depicted` は 2,829件 (6.4%)。ソース別: YouTube 7.9% / TACO 1.3% / COCO 0.7% / LVIS 0.4%
  （目視サンプル確認済み: アニメ・イラスト・画面上の描写を正しく検出）

## 3. これまでの処理履歴（2026-06-27〜28）

1. **TACO重複の dedup / リーク解消** — 同一画像263重複を統合、train↔test 跨ぎ90件のリークを解消
   （[dedup_merge.py](dedup_merge.py)。dedup 前バックアップは検証後 7/11 に削除済み）。
2. **YouTube CC データ収集** — CC絞り込み検索→throttle付きDL→SAM3 "bottle"判定→CLIP多様性dedup。
   10 pod フリート（[runpod/_yt_fleet.py](runpod/_yt_fleet.py)）。13,571枚→pod間グローバルdedup→**9,426枚**採用
   （[collect_youtube.py](collect_youtube.py) / [merge_youtube_fleet.py](merge_youtube_fleet.py)）。
3. **本体へ統合** — YouTubeを `_sam3merge` の all/train/val/test へ追記。split は**動画単位**で割当て
   （近接フレームのリーク防止、[merge_youtube.py](merge_youtube.py)。統合前バックアップは検証後 7/11 に削除済み）。
4. **部位分離 B（part-min=64）** — マージ後データで bottle→cap/label を SAM3 分離。
   **30 pod × 3 worker = 90シャード**を RunPod 並列実行（[runpod/_fleet.py](runpod/_fleet.py) /
   [segment_parts.py](segment_parts.py) / [merge_parts.py](merge_parts.py)）。cap/label 計 80,451 parts 生成。
5. **CVATエクスポート** — 3クラス `_sam3parts` を train/val/test で出力（属性なし旧版のため 7/11 に削除済み。`_sam3full` から要再出力）。
6. RunPod は全 pod 削除済み（課金0）。一時ファイル整理済み。

> RunPod hub-pull はデータセンタ跨ぎで一部 pod が hub 到達不可になることがある（今回30台中8台）。
> 未取得シャードは稼働 pod へ再配分して完走させた。次回は最初から再配分 heal を自動化すると堅い。
> 30台同時 pull は hub の sshd 同時接続上限(~10)に当たるため段階化推奨。

## 4. 機械検査の結果（2026-07-11、[inspect_sam3full.py](inspect_sam3full.py)）

レポート: `datasets/pet_bottle/qa_sam3full/report.json`、
サンプル描画: `qa_sam3full/samples/`（[render_qa_samples.py](render_qa_samples.py)、目視確認済み）。

**問題なし**: ID/参照整合・画像ファイル実在（欠落0/未参照0）・split リーク 0・属性スキーマ違反 0。
`large_bottle_all_unknown` 242 件のうち 240 は iscrowd（属性付与対象外）で正常、実欠落は既知の2件のみ。

**機械修正 適用済み（[fix_sam3full.py](fix_sam3full.py)、2026-07-11。アノテ 200,872 → 182,942）**:
| 問題 | 処理 |
|---|---|
| parts の親リンクずれ 20,949 | crop 検出が隣のボトルの cap/label を取得していた。幾何包含（≥0.5）で正しい bottle へ再リンク |
| parts の二重マスク | 同一画像・同クラス IoU≥0.9 の NMS で 4,472 除去 + 同一親・同クラス IoU≥0.3 の NMS で 7,860 除去（1 crop から cap/label 各 top-1 しか出ないため、同一親の複数 parts は再リンクで合流した重複） |
| 浮遊 parts（ゴミ検出） 5,486 | どの bottle にも包含 <0.5（カップ麺の縁を cap 誤認等）。削除 |
| bottle 二重登録 106 | LVIS 由来の同一ボトル二重アノテ。面積大を残し dedup（属性・parts は引き継ぎ） |
| 画像外の不正アノテ 6 | TACO 由来の負幅 bbox（polygon が完全に画像外）。削除 |

さらに同日、属性矛盾も機械処理（VLM と SAM3 のどちらが正しいか判定不能なため削除ではなく中立化）:
| 問題 | 処理 |
|---|---|
| 属性矛盾（uncapped なのに cap_color/cap part あり 8,380、unlabeled なのに label_color/label part あり 6,852） | presence と color を **unknown 化**（誤教師信号を除去、マスクは温存）。矛盾リストは `qa_sam3full/report.json`（修正前）に残っており CVAT サンプル検品可能 |
| 散在マルチポリゴン bottle 49（1アノテに離れた複数本、COCO 由来） | **iscrowd=1 化**（ignore 領域として学習・評価から除外。iscrowd 計 333 = RLE 284 + 49） |

修正前ファイルの退避: セッション scratchpad `sam3full_prefix_backup/`（一時領域なので恒久保存はしない）。
再検査（`qa_sam3full/report_after_fix.json`）で機械修正対象カテゴリは全て 0 件を確認。修正はべき等。

**CVAT 検品対象（機械修正不可。件数は修正後）**:
- 同一 bottle に離れた cap 複数（IoU<0.3）: 1,474 件。本物+ゴミ、積み重なった別ボトルのキャップ等が混在。
- unknown 化した属性矛盾（上表）のサンプル検品（修正前 report.json の一覧から）。
- bbox⇔seg ずれ >8px: 262 件 / cap・label の sliver マスク 3 件（軽微）。

## 5. 未了 / 次の候補
- [x] **属性(VLM)をマージ後データに付与** — 2026-07-11 完了（30B-A3B bf16、10属性、44,332個体）
- [x] **parts + attrs を単一アノテに統合（`_sam3full`）＋旧ブランチ・旧zip・バックアップ整理** — 2026-07-11 完了
- [x] **機械的品質検査 + 目視サンプル確認** — 2026-07-11 完了（→ §4）
- [x] **機械修正の適用**（親リンク再割当 / parts NMS / 浮遊 parts 削除 / 不正 bbox 除去 /
      属性矛盾の unknown 化 / 散在 bottle の iscrowd 化）— 2026-07-11 完了（→ §4）
- [x] **CVAT 検品準備** — 2026-07-11 完了: cvat_labels_parts.json を10属性対応に更新（旧 cvat_labels.json
      は削除）、`cvat/` に review_multicap(1,136枚) / review_attr(292枚) / val / test の zip を出力。
      手順は [README_CVAT.md](README_CVAT.md)。旧 CVAT プロジェクト（6月 parts 版）は UI 上で削除してよい
- [x] **SAM3 マスクリファイン** — 2026-07-11 完了。全マスク（>=24px 非crowd 155,509）を既存 bbox の
      box プロンプトで再生成（ID/属性/親子関係は不変、`seg_refined: true` 付与）。
      RunPod **H100 80GB ×8**（$2.99/h、16シャード、推論 ~15分）で分散実行。
      155,241 更新 / 268 は安全ガードで旧マスク温存。旧マスクとの IoU 分布:
      >=0.9 58% / 0.7-0.9 38% / <0.5 0.7%（=輪郭の精密化が主で、すり替わりなし）。
      変化の大きい上位500件は `qa_sam3full/refine_low_iou.json`。
      適用後に fix_sam3full を再実行し新 bbox で整合を取り直し（bottle 重複 69 / 浮遊 220 /
      再リンク 406 / NMS 334 を追加除去）。
      仕様・手順: [runpod/README_RUNPOD.md](runpod/README_RUNPOD.md) §0a、
      [refine_masks.py](refine_masks.py) / [merge_refined.py](merge_refined.py)
- [ ] CVAT で目視確認・補正（zip はリファイン済みで再出力済み。優先: review_multicap(1,109枚) →
      review_attr(292枚)。fill_level unknown 58%、depiction 境界例）→ 補正の書き戻し
- [x] **モデル学習の準備** — 2026-07-11 完了。`instances_*_trainready.json`（depicted 2,826個体+partsを
      iscrowd=1 で学習除外 = 「扱い」は ignore で決定）、RTMDet-Ins-s 3クラス config
      （[mmdet_configs/rtmdet-ins_s_pet_bottle.py](mmdet_configs/rtmdet-ins_s_pet_bottle.py)、パース検証済み）、
      RunPod H100 ランブック（[runpod/README_RUNPOD.md](runpod/README_RUNPOD.md) §0b）
- [x] **RTMDet-Ins-s 学習** — 2026-07-11 完了（8×H100、60epoch、run 名 `h100x8_60e_lr2.5e-4`）。
      **test: bbox_mAP 0.376 / segm_mAP 0.352**（val: 0.382 / 0.363、過学習なし。
      対 SAM3 疑似ラベルの再現度であり人手 GT ではない）。
      best ckpt: `work_pet_bottle/best_coco_segm_mAP_epoch_60.pth`、学習曲線: `mlruns/`
      （`mlflow ui --backend-store-uri "file:train/segmentation/mlruns"`）。
      ⚠️ 知見: COCO 事前学習からのファインチューンで lr>5e-4 は val 崩壊
      （train loss は正常のまま SyncBN running 統計が追従不能）→ **lr 2.5e-4 が正解**。
      8×H100 実行時の環境罠4件（NCCL NVLS / mmcv sm_90 / MLflow file store / iscrowd 欠落）は
      runpod/run_train.sh・setup_train.sh に対策済み
- [x] 学習済みモデルの ONNX 化（mmdeploy、`make rtmdet-onnx` → rtmdet_ins.onnx）→ アプリ組み込み・
      Android 実機動作確認済み（2026-07-11。詳細は TRAINING_LOG.md §5-6）
- [ ] 精度改善の候補: CVAT 検品の反映後に再学習 / rtmdet-ins_m へのスケールアップ / 入力解像度調整

## 6. CVATで見るには
検品セット出力済み: `datasets/pet_bottle/cvat/{review_multicap,review_attr,val,test}_cvat_coco.zip`。
ラベル定義は [cvat_labels_parts.json](cvat_labels_parts.json)（bottle 10属性対応済み）を CVAT の
Raw に貼付。手順・検品観点・書き戻しは [README_CVAT.md](README_CVAT.md) を参照。
