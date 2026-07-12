import 'dart:typed_data';

import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';

/// RTMDet-Ins (mmdeploy export) を実行し、インスタンスマスク+枠の RGBA
/// オーバーレイを生成する。
///
/// I/O 契約（model/rtmdet/export_rtmdet.sh で SIZE=320 エクスポート）:
///   input  : float32 [1, 3, S, S]  NCHW、**BGR** 順・mean/std 正規化
///            (mean = 103.53, 116.28, 123.675 / std = 57.375, 57.12, 58.395)
///   dets   : float32 [1, K, 5]  (x1, y1, x2, y2, score) 入力解像度スケール
///   labels : int64   [1, K]
///   masks  : float32 [1, K, S, S]  インスタンスマスク（0..1）
class Detector {
  // scoreThreshold の下限は ONNX 側の内部閾値（export_rtmdet.sh の SCORE_THR=0.25）。
  // それ未満に下げたい場合は再エクスポートが必要。上げる分にはここだけでよい
  // （0.40 は誤検知抑制のための実運用値）。
  Detector({this.inputSize = 320, this.scoreThreshold = 0.40});

  /// モデル入力解像度（エクスポート時の SIZE と一致させること）。
  final int inputSize;
  final double scoreThreshold;

  static const String _assetPath = 'assets/models/rtmdet_ins.onnx';
  static const String _inputName = 'input';

  // RTMDet の data_preprocessor と同値（BGR 順、bgr_to_rgb=False）。
  static const List<double> _meanBgr = [103.53, 116.28, 123.675];
  static const List<double> _stdBgr = [57.375, 57.12, 58.395];

  static const int _fillAlpha = 110;
  // クラス色: bottle / cap / label
  static const List<List<int>> _colors = [
    [235, 64, 64], // bottle: 赤
    [64, 144, 255], // cap: 青
    [64, 220, 120], // label: 緑
  ];

  OnnxRuntime? _ort;
  OrtSession? _session;
  bool get isReady => _session != null;

  /// 直近フレームの検出数（クラス別）。UI 表示用。
  List<int> lastCounts = [0, 0, 0];

  /// 検出数がこれを超えるフレームはマスク転送をスキップして枠のみ描画する
  /// （masks テンソルの platform channel 転送が支配的コストのため）。
  static const int _maskFetchLimit = 15;

  Future<void> init() async {
    _ort = OnnxRuntime();
    if (kIsWeb) {
      // Web ではプラグインがパスをそのまま ort-web の fetch に渡すため、
      // Flutter web の実配信パス（assets/<アセットキー>）を明示する必要がある。
      _session = await _ort!.createSession('assets/$_assetPath');
    } else {
      _session = await _ort!.createSessionFromAsset(_assetPath);
    }
  }

  /// [rgba] は inputSize×inputSize の RGBA バッファ（呼び出し側でリサイズ済み）。
  /// 戻り値: 同解像度の RGBA オーバーレイ（背景透明、マスク塗り+枠線）。
  Future<Uint8List> run(Uint8List rgba) async {
    final session = _session;
    if (session == null) {
      throw StateError('Detector not initialized');
    }

    final int s = inputSize;
    final int plane = s * s;

    // --- 前処理: RGBA(HWC) -> BGR float NCHW 正規化 ---
    final input = Float32List(3 * plane);
    for (int p = 0; p < plane; p++) {
      final int base = p * 4;
      final double r = rgba[base].toDouble();
      final double g = rgba[base + 1].toDouble();
      final double b = rgba[base + 2].toDouble();
      input[p] = (b - _meanBgr[0]) / _stdBgr[0]; // ch0 = B
      input[plane + p] = (g - _meanBgr[1]) / _stdBgr[1]; // ch1 = G
      input[2 * plane + p] = (r - _meanBgr[2]) / _stdBgr[2]; // ch2 = R
    }

    final inputTensor = await OrtValue.fromList(input, [1, 3, s, s]);
    Map<String, OrtValue>? outputs;
    try {
      outputs = await session.run({_inputName: inputTensor});
      final dets =
          (await outputs['dets']!.asFlattenedList()).cast<num>();
      final labels =
          (await outputs['labels']!.asFlattenedList()).cast<num>();

      final int k = labels.length;
      // 閾値を超える検出（スコア降順で並んでいる）
      final accepted = <int>[];
      for (int i = 0; i < k; i++) {
        if (dets[i * 5 + 4].toDouble() < scoreThreshold) continue;
        final int cls = labels[i].toInt();
        if (cls < 0 || cls >= _colors.length) continue;
        accepted.add(i);
      }

      // 密集シーンではマスク転送（支配的コスト）をスキップし枠のみ描画
      List<num>? masks;
      if (accepted.length <= _maskFetchLimit) {
        masks = (await outputs['masks']!.asFlattenedList()).cast<num>();
      }

      final overlay = Uint8List(plane * 4); // 透明で初期化
      final counts = [0, 0, 0];

      for (final i in accepted) {
        final int cls = labels[i].toInt();
        counts[cls]++;
        final color = _colors[cls];

        if (masks != null) {
          final int mOff = i * plane;
          for (int p = 0; p < plane; p++) {
            if (masks[mOff + p].toDouble() < 0.5) continue;
            final int o = p * 4;
            // 既に塗られていたら上書きしない（先勝ち=スコア降順）
            if (overlay[o + 3] != 0) continue;
            overlay[o] = color[0];
            overlay[o + 1] = color[1];
            overlay[o + 2] = color[2];
            overlay[o + 3] = _fillAlpha;
          }
        }

        final int x1 = dets[i * 5].toDouble().clamp(0, s - 1).toInt();
        final int y1 = dets[i * 5 + 1].toDouble().clamp(0, s - 1).toInt();
        final int x2 = dets[i * 5 + 2].toDouble().clamp(0, s - 1).toInt();
        final int y2 = dets[i * 5 + 3].toDouble().clamp(0, s - 1).toInt();
        _drawRect(overlay, s, x1, y1, x2, y2, color);
      }
      lastCounts = counts;
      return overlay;
    } finally {
      await inputTensor.dispose();
      if (outputs != null) {
        for (final v in outputs.values) {
          await v.dispose();
        }
      }
    }
  }

  void _drawRect(
      Uint8List img, int s, int x1, int y1, int x2, int y2, List<int> c) {
    void px(int x, int y) {
      if (x < 0 || y < 0 || x >= s || y >= s) return;
      final int o = (y * s + x) * 4;
      img[o] = c[0];
      img[o + 1] = c[1];
      img[o + 2] = c[2];
      img[o + 3] = 255;
    }

    for (int x = x1; x <= x2; x++) {
      px(x, y1);
      px(x, y2);
    }
    for (int y = y1; y <= y2; y++) {
      px(x1, y);
      px(x2, y);
    }
  }

  Future<void> dispose() async {
    await _session?.close();
    _session = null;
  }
}
