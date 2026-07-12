# bottle-seg-lite — リアルタイム・ペットボトル・インスタンスセグメンテーション

カメラ映像に ONNX モデルの検出・セグメンテーション結果をリアルタイム重畳する
**Flutter（Web / Android）** アプリと、その自作モデル（RTMDet-Ins-s、bottle/cap/label 3クラス）の
**データセット作成〜学習〜ONNX 化パイプライン**一式です。開発は Docker で完結します。

- **映像は止まらない**: カメラ映像はネイティブ層で再生され、Dart/Flutter とは独立して描画されます。
  推論は非同期ループで動き、前フレームの推論が終わるまで新しいフレームは**スキップ**されるため、
  プレビューがモデルを待つことはありません。
- **モデルは差し替え可能**: デモ用の `LR-ASPP MobileNetV3-Large`（`make model`）と、
  自作データセットで学習した `RTMDet-Ins-s`（`make rtmdet-onnx`、要学習済み ckpt）の2系統。

## 構成

```
bottle-seg-lite/
├─ docker-compose.yml        # model/rtmdet-onnx/seg/apk(ツール) と web(devサーバ)
├─ docker-compose-cvat.yml   # アノテーション検品用 CVAT（make cvat-up）
├─ Makefile                  # make model / make up / make apk など
├─ model/                    # ONNX エクスポート
│  ├─ export_model.py        #  デモ用 LR-ASPP (input[1,3,S,S] → output[1,21,S,S])
│  └─ rtmdet/                #  学習済み RTMDet-Ins-s の ONNX 化（mmdeploy、GPU 必須）
├─ app/                      # Flutter アプリ（Web + Android）
│  ├─ web/index.html         # onnxruntime-web の <script> を読み込み
│  └─ lib/
│     ├─ camera_view_web.dart    # getUserMedia + 非ブロッキング推論ループ
│     ├─ camera_view_mobile.dart # camera プラグイン（YUV420→RGBA+回転補正）
│     ├─ detector.dart           # RTMDet-Ins 前処理/推論/後処理
│     ├─ segmenter.dart          # デモ用セマンティックセグ（argmax→色マスク）
│     └─ overlay_paint.dart      # マスク・枠の重畳描画
└─ train/segmentation/       # データセット作成〜学習パイプライン（下記）
```

## 学習パイプライン（train/segmentation/）

pet_bottle データセット（21,612枚 / COCO+LVIS+TACO+YouTube CC、SAM3 による 3クラスマスク +
Qwen3-VL による bottle 属性10種）の作成と、RTMDet-Ins-s の学習（test segm_mAP 0.352）。
データセット本体・学習成果物は git 管理外（ローカル/クラウドのみ）。

- データセット仕様: [train/segmentation/DATASET.md](train/segmentation/DATASET.md)
- 作業ログ・経緯: [train/segmentation/DATASET_STATUS.md](train/segmentation/DATASET_STATUS.md)
- 学習記録と知見（lr/NCCL/mmcv のハマりどころ）: [train/segmentation/TRAINING_LOG.md](train/segmentation/TRAINING_LOG.md)
- SAM3 ローカル実行: [train/segmentation/README_SAM3_local.md](train/segmentation/README_SAM3_local.md)
- CVAT 検品: [train/segmentation/README_CVAT.md](train/segmentation/README_CVAT.md)
- RunPod フリート運用: [train/segmentation/runpod/README_RUNPOD.md](train/segmentation/runpod/README_RUNPOD.md)

シークレットは `.env` に置く（`.env.example` をコピーして作成。git 管理外）。

## 必要なもの

- Docker / Docker Compose
- カメラ付き端末と、`localhost` でアクセスできるブラウザ（Chrome/Edge 推奨）

> getUserMedia は**セキュアコンテキスト**が必要です。`http://localhost:8080`（localhost）は
> セキュア扱いなので HTTPS なしで動きます。別マシンの IP で開く場合は HTTPS が必要です。

## 使い方

```bash
# 1) モデルを ONNX にエクスポート（app/assets/models/seg.onnx を生成）。初回のみ。
make model

# 2) 開発サーバ起動（初回はイメージ build + pub get で数分かかります）
make up
```

起動後、ブラウザで **http://localhost:8080** を開き、カメラ許可を与えてください。
左上に状態と 1フレームあたりの推論時間(ms)が表示されます。

停止は `Ctrl-C` もしくは別ターミナルで `make down`。

`make` を使わない場合:

```bash
docker compose --profile tools run --rm model   # = make model
docker compose up --build web                    # = make up
```

## 仕組み（要点）

- `camera_view.dart`
  - `getUserMedia` で `MediaStream` を取得し `<video>` に流す → `HtmlElementView` で表示。
  - オフスクリーン `<canvas>` に `drawImage` で**モデル入力解像度(256²)へ縮小**して描画し、
    `getImageData` で RGBA を取得。
  - `_loop()` が非同期に回り、`_running` フラグで多重実行を防止（=フレームスキップ）。
  - 推論結果のマスク(RGBA)を `decodeImageFromPixels` で `ui.Image` 化し、`CustomPaint` で
    `object-fit: cover` に合わせて重畳。
- `segmenter.dart`
  - 前処理: RGBA(0-255) → NCHW Float32、ImageNet 正規化。
  - 推論: `flutter_onnxruntime` の `OrtValue.fromList` / `session.run`（Web では index.html の
    `onnxruntime-web` を利用）。
  - 後処理: 21ch のうち `argmax` を取り、背景以外を VOC パレット色で半透明に塗る。
- `web/index.html` に
  `https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/ort.min.js` を読み込み（必須）。

## 自作モデルへの差し替え方

1. ONNX の入出力を以下に合わせる（または `lib/segmenter.dart` の定数を変更）:
   - 入力名 `input`, shape `[1,3,S,S]`, NCHW float32
   - 出力名 `output`, shape `[1,numClasses,S,S]`（チャネル方向 argmax で分類）
2. `model/export_model.py` を置き換えて `make model` で `seg.onnx` を再生成、
   もしくは既存の `.onnx` を `app/assets/models/seg.onnx` に置くだけでも可。
3. クラス数・前処理（mean/std）・配色が違う場合は `segmenter.dart` を調整。

## パフォーマンス改善メモ

WASM の単一スレッド実行だと UI スレッドにジャンクが出ることがあります（ネイティブ `<video>` 自体は
止まりません）。高速化したい場合:

- **マルチスレッド WASM**: 配信サーバで以下のヘッダを付与（SharedArrayBuffer 有効化）:
  `Cross-Origin-Opener-Policy: same-origin` / `Cross-Origin-Embedder-Policy: require-corp`
- **WebGPU EP**: `onnxruntime-web` の WebGPU 実行プロバイダを利用（対応ブラウザ）。
- 入力解像度を下げる（`Segmenter(inputSize: 192)` など）。
```
