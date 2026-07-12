# ローカルパッチ版 flutter_onnxruntime 1.8.0

pub.dev の flutter_onnxruntime 1.8.0 のコピー。`app/pubspec.yaml` の
`dependency_overrides` からパス参照される。

## パッチ内容（lib/web/flutter_onnxruntime_web_plugin.dart のみ）

Web 実装の Dart⇔JS TypedArray 変換が 1 要素ずつの interop 呼び出し
（`getProperty(i.toString())` / `Array.from` 経由）になっており、
416×416×K のマスクテンソルで 100ms 超を消費していたため、
一括変換に置き換えた。`LOCAL PATCH` コメントで検索可能。

- `_convertToTypedArray`: 入力が `Float32List`/`Int32List`/`Uint8List` の
  場合は `.toJS` で直接 JS TypedArray 化（中間 JS Array を作らない）
- `getOrtValueData`: float32/int32/uint8 テンソルは
  `(jsData as JSFloat32Array).toDart` 等で一括変換
- `createJsSessionOptions`: WEB_NN は文字列 `'webnn'` だと deviceType が
  既定 `'cpu'` になるため、`{name: 'webnn', deviceType: 'gpu'}` を渡す

int64（labels、要素数が少ない）と bool/string は元実装のまま。

## パッチ内容（android/build.gradle）

- `onnxruntime-android` 1.22.0 → 1.27.0（Web の ort-web と同一バージョン。
  1.22 は XNNPACK EP が本プロジェクトのモデルでセッション作成中に SIGABRT）

## パッチ内容（android/.../FlutterOnnxruntimePlugin.kt）

- `NNAPI`: `addNnapi(EnumSet.of(NNAPIFlags.USE_FP16))` — fp16 実行を許可
  （GPU/NPU の実効スループット向上）
- `XNNPACK`: 独自スレッドプールが未指定だと 1 スレッドになり CPU EP より
  遅くなるため、セッションの `intraOpNumThreads` を
  `intra_op_num_threads` として引き継ぐ（未指定時 4）

## 更新時の注意

パッケージを上げる場合は pub cache から新版をここへコピーし直し、
上記 2 箇所のパッチを再適用すること。
