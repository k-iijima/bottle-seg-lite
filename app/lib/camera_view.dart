/// プラットフォーム別のカメラ+検出オーバーレイ実装を切り替える。
/// - モバイル/デスクトップ (dart:io あり): camera プラグイン実装
/// - Web: getUserMedia + <video> 実装
export 'camera_view_mobile.dart' if (dart.library.js_interop) 'camera_view_web.dart';
