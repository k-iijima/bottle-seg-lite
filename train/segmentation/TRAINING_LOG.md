# RTMDet-Ins-s 学習記録（2026-07-11）

bottle データセット（`_sam3full` → `_trainready`）での初回学習の結果・手順・知見。
データセット側の経緯は [DATASET_STATUS.md](DATASET_STATUS.md)、実行手順の正は
[runpod/README_RUNPOD.md](runpod/README_RUNPOD.md) §0b。

## 1. 結果

| 指標 | val (1,869枚) | **test (2,133枚)** |
|---|---|---|
| bbox_mAP | 0.382 | **0.376** |
| bbox_mAP_50 | 0.545 | 0.541 |
| segm_mAP | 0.363 | **0.352** |
| segm_mAP_50 | 0.539 | 0.526 |
| segm_mAP (small/med/large) | 0.234 / 0.523 / 0.559 | 0.228 / 0.493 / 0.533 |

- **評価の正解は SAM3 生成アノテーション**（`_sam3full`、人手 GT なし）。上記 mAP は
  SAM3 疑似ラベルの再現度であり、人手正解に対する真の精度とは乖離しうる。
- val→test の劣化がほぼなく**過学習なし**。伸びしろは小物体（segm 0.228）。
- 学習曲線: ep5 0.202 → ep25 0.300 → ep45 0.336 → ep50(増強オフ) 0.341 → **ep60 0.363**。
  最後まで単調改善（60 epoch では飽和しきっていない）。
- 成果物: `work_pet_bottle/best_coco_segm_mAP_epoch_60.pth`（EMA重み）、
  MLflow 記録 `mlruns/`（`mlflow ui --backend-store-uri "file:train/segmentation/mlruns"`、
  実験 `pet_bottle_rtmdet_ins`、run `h100x8_60e_lr2.5e-4`）。

## 2. 構成

| 項目 | 値 |
|---|---|
| モデル | RTMDet-Ins-s（mmdetection v3.3.0）、COCO 300e 事前学習から fine-tune |
| クラス | bottle / cap / label（3クラス + bottle 10属性 ※属性は学習未使用） |
| データ | `instances_*_trainready.json`（depicted 2,826個体+parts は iscrowd=1 で除外） |
| スケジュール | 60 epoch、warmup 300 iter、cosine 減衰(ep30-60)、最後10epochで Mosaic/MixUp オフ |
| バッチ / lr | 32×8GPU=256 / **2.5e-4**（下記知見）、AdamW、AMP |
| ハード | RunPod 8×H100 80GB SXM（$23.92/h）、学習 ~40分 + val/test |
| 記録 | MLflowVisBackend（file ストア） |

## 3. 手順（再現用）

```bash
# ローカル
python make_train_anns.py --data-root datasets/bottle
docker compose --profile tools run --rm seg bash runpod/package_train_inputs.sh
# pod（8xH100、RunPod PyTorch イメージ、22/tcp、disk 100GB）へ scp で転送・展開後:
bash setup_train.sh          # venv: torch2.1cu121 + mmcv2.1(sm_90ソースビルド) + mmdet3.3
GPUS=8 bash run_train.sh --smoke                 # 1 epoch 配線確認
GPUS=8 bash run_train.sh visualizer.vis_backends.1.run_name=<run名>
# 成果物: /workspace/train_outputs.tar.gz（ckpt+ログ+mlruns）→ scp 回収 → pod 削除
```

## 4. 知見（ハマりどころ。スクリプトに対策反映済み）

1. **lr が最重要**: COCO 事前学習からの fine-tune で lr 0.004（公式 from-scratch 値）や
   0.001 は **train loss が正常に下がったまま val だけ崩壊**（0.24→0.006）。
   SyncBN の running 統計が高 lr の重み移動に追従できないのが原因。**2.5e-4 で解決**
   （run_train.sh の既定値化）。warmup 1000 iter は 69 iter/epoch では長すぎ → 300 に短縮。
2. **NCCL**: RunPod コンテナでは 3GPU 以上で NVLS が
   `operation cannot be performed in the present state` で死ぬ → `NCCL_NVLS_ENABLE=0` 必須。
   P2P も 8rank で不安定 → `NCCL_P2P_DISABLE=1`（SHM 経由）。2GPU では再現しないので注意。
3. **mmcv**: openmmlab 配布 wheel（cu121/torch2.1 の 2.1.0/2.2.0 とも）は **sm_90 非対応**
   （NMS で `no kernel image`）→ `TORCH_CUDA_ARCH_LIST=9.0` でソースビルド必須（ninja で ~10分）。
4. **MLflow 3.x**: file ストアはオプトイン制 → `MLFLOW_ALLOW_FILE_STORE=true`。
5. **mmdet の COCO ローダは全 ann に iscrowd 必須**（_sam3full に補完済み）。
6. numpy: mlflow 等が 2.x を引き込む → セットアップ最後に必ず 1.26.4 へ再固定。
7. 運用: 複数 pod の DDP は不可（1 pod のマルチ GPU で）。ssh 越し起動は
   `(setsid nohup ... &)`。ローカル→pod の大容量転送は paramiko でなく**ネイティブ scp**。

## 5. ONNX 化 & Android アプリ（2026-07-11 同日完了）

- **ONNX**: `make rtmdet-onnx`（mmdeploy 1.3.1、入力 320×320 static、後処理込み end2end）
  → `app/assets/models/rtmdet_ins.onnx`（43MB）。デスクトップ CPU 推論 ~20ms。
  ⚠️ mmdeploy の rtmdet-ins エクスポートは **CUDA 必須**（CPU だと device='cuda' 直書きで死ぬ）。
  ⚠️ コンテナ内 onnxruntime は WSL2 の execstack 制限で import 不可 → 検証は Windows ローカルで。
- **ONNX I/O**: input `input` [1,3,320,320]（**BGR**・mean 103.53/116.28/123.675・std 57.375/57.12/58.395）
  → `dets` [1,K,5](x1y1x2y2,score) / `labels` [1,K] / `masks` [1,K,320,320]（K は動的）。
- **Flutter アプリ**: `app/lib/detector.dart`（後処理+オーバーレイ）、カメラ層は
  camera_view_{web,mobile}.dart に分離（mobile は camera プラグイン、YUV420→RGBA+回転補正）。
  `make apk` で Android ビルド → `app/build/app/outputs/flutter-apk/app-release.apk`（138MB、全ABI同梱）。

## 6. 実機動作（Xperia SOG10、2026-07-11）

- **動作確認済み**: カメラプレビューにインスタンスマスク+枠がリアルタイム重畳、**200-400ms/frame**。
- 実機で踏んだ問題と対策（すべて反映済み）:
  1. R8 が onnxruntime の Java クラスを削除 → JNI SIGABRT。`proguard-rules.pro` で keep 必須
  2. ビルドごとに debug keystore が変わり再インストール不可 → compose の `android_keys` ボリュームで永続化
  3. マスク位置ズレ → 入力を「フレーム全体 squash」にし、描画側でフレーム実アスペクトの cover 変換
     （overlay_paint.dart の MaskPainter(srcAspect)）でプレビューと座標系を一致させた
  4. 進行性の遅延（600→1780ms）→ 入力 256 化 + 検出上限 K=10（マスク転送量 1/10）+
     推論間隔 150ms 下限（GC 圧・熱スロットリング対策）
- APK インストール時の Play プロテクト警告はストア外+デバッグ署名による正常な挙動。

## 7. 次の一手

- [ ] さらなる高速化（必要なら）: ORT スレッド数/XNNPACK・NNAPI EP、入力 192、モデル量子化
- [ ] 精度改善: CVAT 検品反映後の再学習 / rtmdet-ins_m / 高解像度入力 / epoch 増
- [ ] APK スリム化: `flutter build apk --split-per-abi`（arm64 のみで ~70MB）

## 8. ボトル属性分類器(2段目)の学習(2026-07-12、2×H100 ~15分)

トラック中のボトルクロップに10属性(DATASET.md §5)を付与する軽量マルチヘッド分類器。
教師は `_sam3full` の Qwen3-VL 疑似ラベル → **精度は「対疑似ラベル再現度」**。

- 実装: [train_attr_cls.py](train_attr_cls.py)(`extract` でクロップ抽出→`train` で学習)。
  クロップは bbox+15% pad・長辺>=96px のボトル 43,477個(train 38,029 / val 2,469 / test 2,979)を
  ローカル抽出して転送(283MB。画像全体4GBを送らない)。
- モデル: torchvision MobileNetV3(ImageNet事前学習) + 共有ネック512 + 属性別10ヘッド。
  入力128×128(アスペクト潰し=ランタイム規約)、unknown は ignore_index、クラス重み sqrt逆頻度、
  bf16 AMP、30 epoch、AdamW 3e-4 cosine。val 平均macro-F1 でベスト選択。
- RunPod: [runpod/_attrcls_pod.py](runpod/_attrcls_pod.py) で 1 pod(H100×2)作成、
  small/large を GPU 並列で同時学習。package→seed→smoke→本走→回収→term の全所要 ~30分。
  ⚠️ Windows で作った tar は pod 側で `--no-same-owner` を付けて展開(chown エラー)。
- 結果(test、対疑似ラベル):

  | arch | mean macro-F1 | 強い属性 | 弱い属性 |
  |---|---|---|---|
  | mobilenet_v3_small(5MB/デプロイ用) | **0.681** | depiction .865 / label .845 / orientation .791 | fill_level .485 / material .522(glass/can混同+other僅少) |
  | mobilenet_v3_large(14MB/参照) | **0.709** | 同傾向 | 同傾向 |

- 成果物: `work_attr/<arch>/best.pth · metrics.json · history.json · attr_cls_*.onnx · onnx_meta.json`
  (ONNX は検出器と同じ uint8 NHWC RGBA [1,128,128,4] 入力埋め込み、出力 logits [1,41]。
  ORT 検算済み max diff ~5e-3)
- 次: アプリ統合(トラック単位の非同期推論+時間集約)、fill_level/material の教師見直し

### 8b. MobileNetV4 比較(2026-07-12、2×H100 ~15分)

timm の V4 conv 系(hybrid は MQA 入りで量子化に不利なので除外)を同条件で学習して比較。
timm の efficientnet 系は `num_classes=0` でも conv_head 込みの `head_hidden_size` 次元が
出てくる(num_features ではない)点に注意。

| arch | ONNX | test meanF1 |
|---|---|---|
| mobilenet_v3_small | 5.0MB | 0.681 |
| mobilenetv4_conv_small | 12.7MB | 0.699 |
| **mobilenet_v3_large** | **13.9MB** | **0.709** |
| mobilenetv4_conv_medium | 36.4MB | 0.712 |

**結論: 同サイズ帯では V3 が V4 に勝つか同等**(13-14MB 帯で v3_large 0.709 ≥ v4_conv_small 0.699、
36MB の v4_conv_medium でも 0.712 と v3_large +0.4pt のみ)。128px 入力・この教師では V4 の優位は
出なかった。**デプロイは v3_small(5MB)のまま**、精度を買うなら v3_large(13.9MB)が最有力。
成果物は work_attr/ に4アーキ分。
