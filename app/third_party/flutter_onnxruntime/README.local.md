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

int64（labels、要素数が少ない）と bool/string は元実装のまま。
Android などネイティブ実装には変更なし。

## 更新時の注意

パッケージを上げる場合は pub cache から新版をここへコピーし直し、
上記 2 箇所のパッチを再適用すること。
