# bottle-seg-lite — だいたいのボトルをざっくり検出セグメンテーションするモデル

カメラ映像に ONNX モデルの検出・セグメンテーション結果をリアルタイム重畳する
**Flutter（Web / Android）** アプリと、その自作モデル（RTMDet-Ins-s、bottle/cap/label 3クラス）の
**データセット作成〜学習〜ONNX 化パイプライン**一式です。開発は Docker で完結します。

**🌐 ライブデモ: https://k-iijima.github.io/bottle-seg-lite/** （カメラ許可が必要。
main への push で GitHub Actions が自動ビルド・デプロイします）

- **映像をとめない**: カメラ映像はネイティブ層で再生され、Dart/Flutter とは独立して描画されます。
  推論は非同期ループで動き、前フレームの推論が終わるまで新しいフレームは**スキップ**されるため、
  プレビューがモデルを待つことはありません。
- **モデル**: 自作データセットで学習した `RTMDet-Ins-s`。Releases からダウンロードするか、
  学習済み ckpt から `make rtmdet-onnx` で ONNX 化して配置します。

## 構成

```
bottle-seg-lite/
├─ docker-compose.yml        # rtmdet-onnx/seg/apk(ツール) と web(devサーバ)
├─ docker-compose-cvat.yml   # アノテーション検品用 CVAT（make cvat-up）
├─ Makefile                  # make up / make apk / make rtmdet-onnx など
├─ model/
│  └─ rtmdet/                #  学習済み RTMDet-Ins-s の ONNX 化（mmdeploy、GPU 必須）
├─ app/                      # Flutter アプリ（Web + Android）
│  ├─ web/index.html         # onnxruntime-web の <script> を読み込み
│  └─ lib/
│     ├─ camera_view_web.dart    # getUserMedia + 非ブロッキング推論ループ
│     ├─ camera_view_mobile.dart # camera プラグイン（YUV420→RGBA+回転補正）
│     ├─ detector.dart           # RTMDet-Ins 前処理/推論/後処理
│     └─ overlay_paint.dart      # マスク・枠の重畳描画
└─ train/segmentation/       # データセット作成〜学習パイプライン（下記）
```

## 学習パイプライン（train/segmentation/）

bottle データセット（21,612枚 / COCO+LVIS+TACO+YouTube CC、SAM3 による 3クラスマスク +
Qwen3-VL による bottle 属性10種）の作成と、RTMDet-Ins-s の学習（test segm_mAP 0.352。
評価の正解も SAM3 生成アノテーションのため、この値は人手正解に対する精度ではなく
**SAM3 疑似ラベルの再現度**）。
データセット本体・学習成果物は git 管理外。

- データセット仕様: [train/segmentation/DATASET.md](train/segmentation/DATASET.md)
- 作業ログ・経緯: [train/segmentation/DATASET_STATUS.md](train/segmentation/DATASET_STATUS.md)
- 学習記録と知見（lr/NCCL/mmcv のハマりどころ）: [train/segmentation/TRAINING_LOG.md](train/segmentation/TRAINING_LOG.md)
- SAM3 ローカル実行: [train/segmentation/README_SAM3_local.md](train/segmentation/README_SAM3_local.md)
- CVAT 検品: [train/segmentation/README_CVAT.md](train/segmentation/README_CVAT.md)
- RunPod フリート運用: [train/segmentation/runpod/README_RUNPOD.md](train/segmentation/runpod/README_RUNPOD.md)

シークレットは `.env` に置く（`.env.example` をコピーして作成。git 管理外）。

## 学習済みモデル（GitHub Releases）

学習済みモデルは [Releases](https://github.com/k-iijima/bottle-seg-lite/releases) からダウンロードできます
（test segm_mAP 0.352 — 対 SAM3 疑似ラベルの再現度。ONNX I/O 仕様はリリースノート参照）。

| ファイル | 配置先 / 用途 |
|---|---|
| `rtmdet_ins.onnx` | `app/assets/models/rtmdet_ins.onnx` に置くと Flutter アプリで推論可能 |
| `best_coco_segm_mAP_epoch_60.pth` | `train/segmentation/work_pet_bottle/` に置くと `make rtmdet-onnx` で再エクスポート可能 |

```bash
gh release download v0.2.0 -R k-iijima/bottle-seg-lite -p rtmdet_ins.onnx -O app/assets/models/rtmdet_ins.onnx
```

## 必要なもの

- Docker / Docker Compose
- カメラ付き端末と、`localhost` でアクセスできるブラウザ（Chrome/Edge 推奨）

> getUserMedia は**セキュアコンテキスト**が必要です。`http://localhost:8080`（localhost）は
> セキュア扱いなので HTTPS なしで動きます。別マシンの IP で開く場合は HTTPS が必要です。

## 使い方

```bash
# 1) 学習済みモデルを配置（初回のみ。上記「学習済みモデル」の gh release download を実行）

# 2) 開発サーバ起動（初回はイメージ build + pub get で数分かかります）
make up
```

起動後、ブラウザで **http://localhost:8080** を開き、カメラ許可を与えてください。
左上に状態と 1フレームあたりの推論時間(ms)が表示されます。

停止は `Ctrl-C` もしくは別ターミナルで `make down`。

`make` を使わない場合:

```bash
docker compose up --build web                    # = make up
```

## 仕組み（要点）

- `camera_view_web.dart`
  - `getUserMedia` で `MediaStream` を取得し `<video>` に流す → `HtmlElementView` で表示。
  - オフスクリーン `<canvas>` に `drawImage` で**モデル入力解像度へ縮小**して描画し、
    `getImageData` で RGBA を取得。
  - `_loop()` が非同期に回り、`_running` フラグで多重実行を防止（=フレームスキップ）。
  - 推論結果のオーバーレイ(RGBA)を `decodeImageFromPixels` で `ui.Image` 化し、`CustomPaint` で
    `object-fit: cover` に合わせて重畳。
- `detector.dart`
  - 前処理: RGBA(0-255) → BGR の NCHW Float32、RTMDet の mean/std 正規化。
  - 推論: `flutter_onnxruntime` の `OrtValue.fromList` / `session.run`（Web では index.html の
    `onnxruntime-web` を利用）。
  - 後処理: `dets`/`labels`/`masks` をスコア閾値で選別し、クラス色のマスク+枠を描画。
- `web/index.html` に
  `https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/ort.min.js` を読み込み（必須）。

## 自作モデルへの差し替え方

1. ONNX の入出力を以下に合わせる（または `lib/detector.dart` の定数を変更）:
   - 入力 `input`: float32 `[1,3,S,S]` NCHW、**BGR** 順・mean/std 正規化
   - 出力 `dets` `[1,K,5]` / `labels` `[1,K]` / `masks` `[1,K,S,S]`（mmdeploy の RTMDet-Ins 形式）
2. `.onnx` を `app/assets/models/rtmdet_ins.onnx` に置く（学習済み ckpt からの
   再エクスポートは `make rtmdet-onnx`）。
3. クラス数・配色・スコア閾値が違う場合は `detector.dart` を調整。

## パフォーマンス（Web）

以下は実装済み（RTX 4060 Laptop 実測: WebGPU 115ms/8fps、WebNN 44ms/20fps）:

- **入力 320・前処理モデル内蔵**: 入力は uint8 RGBA NHWC で canvas の
  getImageData を直渡し（RGBA→BGR・正規化は `embed_preprocess.py` が
  ONNX グラフ先頭に埋め込み）。
- **転送ゼロコピー化**: flutter_onnxruntime の Web 実装が 1 要素ずつの
  interop 変換でボトルネックだったため、vendored パッチで一括変換に修正
  （`app/third_party/flutter_onnxruntime/README.local.md`）。
- **パイプライン化+ボックス外挿**: マスク合成/デコードは次フレームの推論と
  並行実行。枠はベクタ描画で、検出間は直近 2 検出からの線形外挿で 33ms ごと
  に追従更新（上限 300ms）。
- **WebGPU 実行プロバイダ**（既定）: 非対応ブラウザでは ort-web が自動で wasm にフォールバック。
- **WebNN（実験・要ブラウザフラグ）**: chrome://flags で WebNN を有効化して
  メニューから選択。ネイティブ ML ランタイム（Windows は DirectML）直結のため
  WebGPU 比 3〜4 倍速い。将来ブラウザ既定有効になれば昇格予定。
- **マルチスレッド WASM**: GitHub Pages はヘッダを付与できないため
  `web/coi-serviceworker.js` が COOP/COEP を注入して SharedArrayBuffer を有効化
  （初回アクセス時に 1 回自動リロード。Flutter の SW と衝突するためビルドは
  `--pwa-strategy none`）。
- **int8 量子化モデル**（43MB→12MB）: `make rtmdet-onnx` 後に
  `python model/rtmdet/quantize_int8.py` で生成。感度の高い層
  （SE-attention / backbone stage2.1 blocks.0）は除外済み。
- **右上 🎛 メニュー**で fp32/GPU・fp32/WebNN・fp32/CPU・int8/CPU を実行中に切替可能
  （int8×GPU は ort-web の WebGPU が per-channel DequantizeLinear 未対応のため提供しない。
  fp16×WebGPU は速度向上がなく box 座標が壊れるため非提供、変換スクリプトのみ
  `model/rtmdet/convert_fp16.py` に残置）。
  ステータスチップ左端に実行中モードを常時表示（例: `fp32/WebNN`、
  フォールバック時は `fp32/GPU→CPU×8`）。

Android は NNAPI（fp16 許可）→ XNNPACK（4 スレッド）→ CPU の優先割り当て。

さらに下げたい場合は入力解像度を落として再エクスポート
（`export_rtmdet.sh` の `SIZE` と `Detector(inputSize:)` を一致させる）。
