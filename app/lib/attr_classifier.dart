import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';

/// ボトル属性(10種)の定義。train_attr_cls.py の ATTR_CLASSES と同一順序で、
/// ONNX 出力 logits [1,41] のオフセットに直結するため変更禁止
/// (train/segmentation/work_attr/<arch>/onnx_meta.json が正)。
class AttrSchema {
  /// jp は表示用の属性名、jpClasses は classes と同順の表示用対訳
  /// （fill_level の full=満杯 / visibility の full=全体 のように
  /// 同じ英語値でも属性で訳が違うため属性ごとに持つ）。
  static const List<
      ({
        String key,
        String jp,
        List<String> classes,
        List<String> jpClasses
      })> heads = [
    (
      key: 'material',
      jp: '材質',
      classes: ['pet', 'glass', 'can', 'other'],
      jpClasses: ['PET', 'ガラス', '缶', 'その他'],
    ),
    (
      key: 'cap',
      jp: '蓋',
      classes: ['capped', 'uncapped'],
      jpClasses: ['あり', 'なし'],
    ),
    (
      key: 'cap_color',
      jp: '蓋色',
      classes: [
        'none', 'white', 'black', 'blue', 'red', 'green', 'yellow',
        'silver', 'orange', 'transparent'
      ],
      jpClasses: ['なし', '白', '黒', '青', '赤', '緑', '黄', '銀', '橙', '透明'],
    ),
    (
      key: 'label',
      jp: 'ラベル',
      classes: ['labeled', 'unlabeled'],
      jpClasses: ['あり', 'なし'],
    ),
    (
      key: 'label_color',
      jp: 'ラベル色',
      classes: [
        'none', 'white', 'blue', 'green', 'yellow', 'red', 'black',
        'orange', 'brown', 'multicolor'
      ],
      jpClasses: ['なし', '白', '青', '緑', '黄', '赤', '黒', '橙', '茶', '多色'],
    ),
    (
      key: 'fill_level',
      jp: '中身',
      classes: ['empty', 'low', 'half', 'high', 'full'],
      jpClasses: ['空', '少', '半分', '多', '満杯'],
    ),
    (
      key: 'crushed',
      jp: '変形',
      classes: ['intact', 'crushed'],
      jpClasses: ['なし', 'あり'],
    ),
    (
      key: 'visibility',
      jp: '見え方',
      classes: ['full', 'occluded'],
      jpClasses: ['全体', '一部'],
    ),
    (
      key: 'orientation',
      jp: '向き',
      classes: ['upright', 'lying'],
      jpClasses: ['縦', '横'],
    ),
    (
      key: 'depiction',
      jp: '種別',
      classes: ['real', 'depicted'],
      jpClasses: ['実物', '描画'],
    ),
  ];

  static final int logitsLen =
      heads.fold(0, (n, h) => n + h.classes.length);
}

/// トラック内の属性推定の時間集約(softmax の EMA)。属性はオブジェクトに
/// ほぼ静的なので、単発の誤答(ブレ・一時オクルージョン)をならして安定させる。
class AttrAggregate {
  static const double _alpha = 0.3; // 新規観測の重み

  /// 表示に必要な最低信頼度(EMA 後の top1 確率)。未満は '?' 表示。
  static const double confThreshold = 0.5;

  /// 属性ごとの表示閾値。2値ヘッドは top1 が必ず 0.5 以上になり閾値 0.5 が
  /// 無意味（常にどちらかが表示される）なので高めに要求する。
  static double thresholdFor(int nClasses) =>
      nClasses <= 2 ? 0.65 : confThreshold;

  final List<Float64List> _probs = [
    for (final h in AttrSchema.heads) Float64List(h.classes.length),
  ];

  /// ヘッドごとの観測回数（分類器出力と検出器由来の証拠の両方で増える）。
  final List<int> _n = List.filled(AttrSchema.heads.length, 0);

  /// [logits] は AttrClassifier.run の出力(長さ 41)。
  void add(List<double> logits) {
    int off = 0;
    for (var i = 0; i < AttrSchema.heads.length; i++) {
      final n = AttrSchema.heads[i].classes.length;
      // softmax(数値安定化のため max を引く)
      var m = logits[off];
      for (var c = 1; c < n; c++) {
        m = math.max(m, logits[off + c]);
      }
      var sum = 0.0;
      final p = Float64List(n);
      for (var c = 0; c < n; c++) {
        p[c] = math.exp(logits[off + c] - m);
        sum += p[c];
      }
      final ema = _probs[i];
      for (var c = 0; c < n; c++) {
        final v = p[c] / sum;
        ema[c] = _n[i] == 0 ? v : ema[c] * (1 - _alpha) + v * _alpha;
      }
      _n[i]++;
      off += n;
    }
  }

  /// 分類器以外の証拠（検出器の cap/label パーツ有無など）を、[key] ヘッドの
  /// クラス [cls] への one-hot 観測として重み [alpha] で融合する。
  void addObservation(String key, int cls, {double alpha = 0.3}) {
    final i = AttrSchema.heads.indexWhere((h) => h.key == key);
    if (i < 0) return;
    final ema = _probs[i];
    for (var c = 0; c < ema.length; c++) {
      final v = c == cls ? 1.0 : 0.0;
      ema[c] = _n[i] == 0 ? v : ema[c] * (1 - alpha) + v * alpha;
    }
    _n[i]++;
  }

  /// 属性ごとの (値, 信頼度)。信頼度は EMA 後の top1 確率。
  /// 観測が一度もないヘッドは含めない。
  Map<String, ({String value, double conf})> display() {
    final out = <String, ({String value, double conf})>{};
    for (var i = 0; i < AttrSchema.heads.length; i++) {
      if (_n[i] == 0) continue;
      final h = AttrSchema.heads[i];
      var best = 0;
      for (var c = 1; c < h.classes.length; c++) {
        if (_probs[i][c] > _probs[i][best]) best = c;
      }
      out[h.key] = (value: h.classes[best], conf: _probs[i][best]);
    }
    return out;
  }
}

/// ボトルクロップの10属性推定(2段目の軽量分類器)。
///
/// I/O 契約(train_attr_cls.py の export_onnx で生成):
///   input  : uint8   [1, 128, 128, 4]  NHWC RGBA そのまま
///            (RGB 抽出・ImageNet 正規化はモデル内蔵。検出器と同じ流儀)
///   logits : float32 [1, 41]  AttrSchema.heads 順のヘッド別ロジット
///
/// 教師が Qwen3-VL 疑似ラベルのため、確信度は「対疑似ラベル再現度」相当。
class AttrClassifier {
  static const String asset = 'assets/models/attr_cls.onnx';
  static const int inputSize = 128;

  /// 学習データの付与条件(クロップ長辺 >= 96px)に合わせ、これ未満の
  /// クロップ(元解像度換算)では推論しない。
  static const int minCropSide = 96;

  OnnxRuntime? _ort;
  OrtSession? _session;
  bool get isReady => _session != null;

  /// 直近の推論所要時間 [ms](UI 表示用)。
  int lastRunMs = 0;

  Future<void> init() async {
    final ort = _ort ??= OnnxRuntime();
    if (kIsWeb) {
      // 小モデルなので wasm(CPU)で十分。検出器の WebGPU と競合させない。
      _session = await ort.createSession(
        'assets/$asset',
        options: OrtSessionOptions(providers: [OrtProvider.WEB_ASSEMBLY]),
      );
    } else {
      // Android: 検出器(4スレッド)を圧迫しないよう 2 スレッドに抑える。
      _session = await ort.createSessionFromAsset(
        asset,
        options: OrtSessionOptions(
          providers: [OrtProvider.CPU],
          intraOpNumThreads: 2,
        ),
      );
    }
  }

  /// [rgba] は inputSize×inputSize の RGBA バッファ。logits(長さ41)を返す。
  Future<List<double>> run(Uint8List rgba) async {
    final session = _session;
    if (session == null) {
      throw StateError('AttrClassifier not initialized');
    }
    final sw = Stopwatch()..start();
    final input =
        await OrtValue.fromList(rgba, [1, inputSize, inputSize, 4]);
    Map<String, OrtValue>? outputs;
    try {
      outputs = await session.run({'input': input});
      final logits =
          (await outputs['logits']!.asFlattenedList()).cast<num>();
      lastRunMs = sw.elapsedMilliseconds;
      return [for (final v in logits) v.toDouble()];
    } finally {
      await input.dispose();
      if (outputs != null) {
        for (final v in outputs.values) {
          await v.dispose();
        }
      }
    }
  }

  Future<void> dispose() async {
    await _session?.close();
    _session = null;
  }
}

/// [src](srcW×srcH RGBA)を [dstSize]×[dstSize] にバイリニア縮小する
/// (アスペクトは潰す=学習時と同じ規約)。モバイルの属性クロップ用。
Uint8List resizeRgbaBilinear(
    Uint8List src, int srcW, int srcH, int dstSize) {
  final out = Uint8List(dstSize * dstSize * 4);
  final double kx = srcW / dstSize;
  final double ky = srcH / dstSize;
  for (int dy = 0; dy < dstSize; dy++) {
    final double fy = (dy + 0.5) * ky - 0.5;
    final int y0 = fy.floor().clamp(0, srcH - 1);
    final int y1 = (y0 + 1).clamp(0, srcH - 1);
    final double wy = (fy - y0).clamp(0.0, 1.0);
    for (int dx = 0; dx < dstSize; dx++) {
      final double fx = (dx + 0.5) * kx - 0.5;
      final int x0 = fx.floor().clamp(0, srcW - 1);
      final int x1 = (x0 + 1).clamp(0, srcW - 1);
      final double wx = (fx - x0).clamp(0.0, 1.0);
      final int o = (dy * dstSize + dx) * 4;
      for (int c = 0; c < 4; c++) {
        final double top = src[(y0 * srcW + x0) * 4 + c] * (1 - wx) +
            src[(y0 * srcW + x1) * 4 + c] * wx;
        final double bot = src[(y1 * srcW + x0) * 4 + c] * (1 - wx) +
            src[(y1 * srcW + x1) * 4 + c] * wx;
        out[o + c] = (top * (1 - wy) + bot * wy).round().clamp(0, 255);
      }
    }
  }
  return out;
}
