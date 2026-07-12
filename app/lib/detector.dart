import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';

/// 1 検出。座標はモデル入力解像度のピクセル座標。
class Detection {
  Detection({
    required this.cls,
    required this.score,
    required this.rect,
    required this.srcIndex,
  });

  final int cls;
  final double score;
  final ui.Rect rect;

  /// dets/masks テンソル内の元インデックス（マスク参照用）。
  final int srcIndex;

  ui.Offset get center => rect.center;
}

/// runRaw() の結果。masks は [K, S, S] のフラット列（検出過多時は null）。
class InferResult {
  InferResult({
    required this.detections,
    required this.masks,
    required this.inputSize,
  });

  final List<Detection> detections;
  final List<num>? masks;
  final int inputSize;
}

/// RTMDet-Ins (mmdeploy export) を実行し、インスタンスマスク+枠の RGBA
/// オーバーレイを生成する。
///
/// I/O 契約（model/rtmdet/export_rtmdet.sh で SIZE=320 エクスポート、
/// embed_preprocess.py で前処理をグラフに埋め込み済み）:
///   input  : uint8   [1, S, S, 4]  NHWC RGBA そのまま（canvas getImageData /
///            カメラフレーム直渡し。BGR 変換・mean/std 正規化はモデル内蔵）
///   dets   : float32 [1, K, 5]  (x1, y1, x2, y2, score) 入力解像度スケール
///   labels : int64   [1, K]
///   masks  : float32 [1, K, S, S]  インスタンスマスク（0..1）
class Detector {
  // scoreThreshold の下限は ONNX 側の内部閾値（export_rtmdet.sh の SCORE_THR=0.35）。
  // それ未満に下げたい場合は再エクスポートが必要。上げる分にはここだけでよい
  // （0.40 は誤検知抑制のための実運用値）。
  Detector({this.inputSize = 320, this.scoreThreshold = 0.40});

  /// モデル入力解像度（エクスポート時の SIZE と一致させること）。
  final int inputSize;
  final double scoreThreshold;

  /// 切替可能なモデル（int8 は quantize_int8.py、fp16 は convert_fp16.py で生成）。
  static const String fp32Asset = 'assets/models/rtmdet_ins.onnx';
  static const String fp16Asset = 'assets/models/rtmdet_ins_fp16.onnx';
  static const String int8Asset = 'assets/models/rtmdet_ins_int8.onnx';

  static const String _inputName = 'input';

  static const int fillAlpha = 110;
  // クラス色: bottle / cap / label
  static const List<List<int>> colors = [
    [235, 64, 64], // bottle: 赤
    [64, 144, 255], // cap: 青
    [64, 220, 120], // label: 緑
  ];

  OnnxRuntime? _ort;
  OrtSession? _session;
  bool get isReady => _session != null;

  /// 直近フレームの検出数（クラス別）。UI 表示用。
  List<int> lastCounts = [0, 0, 0];

  /// 直近のステージ別所要時間 [ms]。ボトルネック特定用。
  /// ten=入力テンソル生成 / run=session.run /
  /// dets=dets+labels 転送 / masks=masks 転送 / ovl=オーバーレイ合成
  final Map<String, int> lastStageMs = <String, int>{};

  /// 検出数がこれを超えるフレームはマスク転送をスキップして枠のみ描画する
  /// （masks テンソルの転送コスト抑制）。
  static const int _maskFetchLimit = 15;

  Future<void> init() async {
    await load(modelAsset: fp32Asset, preferGpu: true);
  }

  /// モデル・実行プロバイダを（再）ロードする。実行中のセッションは閉じる。
  ///
  /// [preferGpu] は Web のみ有効: WebGPU を優先し、非対応環境では ort-web が
  /// 自動で wasm にフォールバックする。false なら wasm（CPU）固定。
  /// [webNn] は実験用: WebNN (deviceType=gpu) を優先する（要ブラウザフラグ。
  /// navigator.ml がなければ wasm にフォールバック）。
  Future<void> load({
    required String modelAsset,
    bool preferGpu = true,
    bool webNn = false,
  }) async {
    final ort = _ort ??= OnnxRuntime();
    final old = _session;
    _session = null;
    await old?.close();
    if (kIsWeb) {
      // Web ではプラグインがパスをそのまま ort-web の fetch に渡すため、
      // Flutter web の実配信パス（assets/<アセットキー>）を明示する必要がある。
      final options = OrtSessionOptions(
        providers: webNn
            ? [OrtProvider.WEB_NN, OrtProvider.WEB_ASSEMBLY]
            : preferGpu
                ? [OrtProvider.WEB_GPU, OrtProvider.WEB_ASSEMBLY]
                : [OrtProvider.WEB_ASSEMBLY],
      );
      _session = await ort.createSession('assets/$modelAsset', options: options);
    } else {
      // Android: NNAPI（GPU/DSP/NPU へ委譲、fp16 実行許可はプラグイン側で設定）
      // → XNNPACK（SIMD 最適化 CPU）→ CPU の優先順。NNAPI が扱えない op
      // （NMS 等の後処理）は自動的に後続プロバイダへ割り当てられる。
      final options = OrtSessionOptions(
        providers: preferGpu
            ? [OrtProvider.NNAPI, OrtProvider.XNNPACK, OrtProvider.CPU]
            : [OrtProvider.XNNPACK, OrtProvider.CPU],
        intraOpNumThreads: 4,
      );
      _session = await ort.createSessionFromAsset(modelAsset, options: options);
    }
  }

  /// 推論のみ実行し、検出リストとマスク生データを返す（オーバーレイ合成なし）。
  /// [rgba] は inputSize×inputSize の RGBA バッファ（呼び出し側でリサイズ済み）。
  Future<InferResult> runRaw(Uint8List rgba) async {
    final session = _session;
    if (session == null) {
      throw StateError('Detector not initialized');
    }

    final int s = inputSize;
    final sw = Stopwatch()..start();

    // 前処理（RGBA→BGR・正規化）はモデル内蔵。RGBA バッファを直渡しする。
    final inputTensor = await OrtValue.fromList(rgba, [1, s, s, 4]);
    lastStageMs['ten'] = sw.elapsedMilliseconds;
    Map<String, OrtValue>? outputs;
    try {
      sw.reset();
      outputs = await session.run({_inputName: inputTensor});
      lastStageMs['run'] = sw.elapsedMilliseconds;

      sw.reset();
      final dets =
          (await outputs['dets']!.asFlattenedList()).cast<num>();
      final labels =
          (await outputs['labels']!.asFlattenedList()).cast<num>();
      lastStageMs['dets'] = sw.elapsedMilliseconds;

      final int k = labels.length;
      // 閾値を超える検出（スコア降順で並んでいる）
      final detections = <Detection>[];
      final counts = [0, 0, 0];
      for (int i = 0; i < k; i++) {
        final double score = dets[i * 5 + 4].toDouble();
        if (score < scoreThreshold) continue;
        final int cls = labels[i].toInt();
        if (cls < 0 || cls >= colors.length) continue;
        counts[cls]++;
        detections.add(Detection(
          cls: cls,
          score: score,
          rect: ui.Rect.fromLTRB(
            dets[i * 5].toDouble().clamp(0, s - 1),
            dets[i * 5 + 1].toDouble().clamp(0, s - 1),
            dets[i * 5 + 2].toDouble().clamp(0, s - 1),
            dets[i * 5 + 3].toDouble().clamp(0, s - 1),
          ),
          srcIndex: i,
        ));
      }
      lastCounts = counts;

      // 密集シーンではマスク転送（支配的コスト）をスキップし枠のみ描画
      sw.reset();
      List<num>? masks;
      if (detections.length <= _maskFetchLimit) {
        masks = (await outputs['masks']!.asFlattenedList()).cast<num>();
      }
      lastStageMs['masks'] = sw.elapsedMilliseconds;

      return InferResult(detections: detections, masks: masks, inputSize: s);
    } finally {
      await inputTensor.dispose();
      if (outputs != null) {
        for (final v in outputs.values) {
          await v.dispose();
        }
      }
    }
  }

  /// マスク塗りのみの RGBA オーバーレイを合成する（枠は含まない。
  /// Web は枠を CustomPaint 側でベクタ描画し、検出間はボックス外挿する）。
  Uint8List composeMasks(InferResult r) {
    final int s = r.inputSize;
    final int plane = s * s;
    final sw = Stopwatch()..start();
    final overlay = Uint8List(plane * 4); // 透明で初期化

    final masks = r.masks;
    if (masks != null) {
      for (final d in r.detections) {
        final color = colors[d.cls];
        final int mOff = d.srcIndex * plane;
        for (int p = 0; p < plane; p++) {
          if (masks[mOff + p].toDouble() < 0.5) continue;
          final int o = p * 4;
          // 既に塗られていたら上書きしない（先勝ち=スコア降順）
          if (overlay[o + 3] != 0) continue;
          overlay[o] = color[0];
          overlay[o + 1] = color[1];
          overlay[o + 2] = color[2];
          overlay[o + 3] = fillAlpha;
        }
      }
    }
    lastStageMs['ovl'] = sw.elapsedMilliseconds;
    return overlay;
  }

  /// 旧 API（mobile 用）: マスク+枠入りオーバーレイを一括生成する。
  Future<Uint8List> run(Uint8List rgba) async {
    final r = await runRaw(rgba);
    final overlay = composeMasks(r);
    final int s = r.inputSize;
    for (final d in r.detections) {
      _drawRect(overlay, s, d.rect.left.toInt(), d.rect.top.toInt(),
          d.rect.right.toInt(), d.rect.bottom.toInt(), colors[d.cls]);
    }
    return overlay;
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
