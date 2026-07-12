import 'dart:async';
import 'dart:io';
import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';

import 'attr_classifier.dart';
import 'detector.dart';
import 'overlay_paint.dart';
import 'tracking.dart';

/// モバイル(Android/iOS)向け: camera プラグインのプレビューに RTMDet-Ins の
/// 検出オーバーレイ（bottle=赤 / cap=青 / label=緑）を重ねる。
///
/// - モデル入力は**フレーム全体の squash**（正方クロップしない）。オーバーレイは
///   MaskPainter がフレームの論理アスペクトで cover 変換するのでプレビューと一致する。
/// - 推論は前フレーム処理中はスキップ + 最小間隔でスロットリング（発熱対策）。
class CameraSegView extends StatefulWidget {
  const CameraSegView({super.key});

  @override
  State<CameraSegView> createState() => _CameraSegViewState();
}

class _CameraSegViewState extends State<CameraSegView> {
  final Detector _detector = Detector(inputSize: 320);

  /// 推論の最小間隔（発熱・GC 圧の抑制）。NNAPI/XNNPACK 有効化に合わせて
  /// 150ms→66ms（~15fps 上限）に緩和。発熱が問題になれば戻す。
  static const Duration _minInterval = Duration(milliseconds: 66);

  CameraController? _controller;
  ui.Image? _mask;
  double _frameAspect = 1.0; // 回転補正後フレームの 幅/高さ
  bool _running = false;
  bool _disposed = false;
  DateTime _lastStart = DateTime.fromMillisecondsSinceEpoch(0);

  String _status = 'Initializing…';
  int _lastInferMs = 0;
  double _fps = 0;
  DateTime _lastDone = DateTime.fromMillisecondsSinceEpoch(0);

  /// ステージ別所要時間の表示文字列（ボトルネック特定用、Web 版と同形式）。
  String _timing = '';

  /// 検出枠（入力px座標のベクタ描画。trackId 付きはハイライト）。
  List<({ui.Rect rect, int cls, int? trackId})> _boxes = const [];

  /// 直近フレームの検出リスト（タップ選択用）。
  List<Detection> _lastDets = const [];

  // --- タップ追跡（複数ボトル+キャップ/ラベルの固定表示） ---
  final MultiTracker _tracker = MultiTracker();

  // --- 属性分類器（2段目、トラック中のボトルのみ・低頻度） ---
  final AttrClassifier _attrCls = AttrClassifier();

  /// 連続失敗がこの回数に達したら属性推論を止める（成功でリセット）。
  int _attrFails = 0;
  static const int _maxAttrFails = 5;

  /// トラックあたりの属性推論の最小間隔。属性はほぼ静的なので低頻度でよい
  /// （検出が CPU 200-400ms/frame の機種を想定し、Web より長めにとる）。
  static const Duration _attrInterval = Duration(milliseconds: 1000);

  static final List<Color> _palette = [
    for (final c in Detector.colors) Color.fromARGB(255, c[0], c[1], c[2]),
  ];

  @override
  void initState() {
    super.initState();
    _setup();
  }

  Future<void> _setup() async {
    try {
      _setStatus('Loading model…');
      // flutter_onnxruntime は asset を temp にコピーして再利用するため、
      // APK 更新でモデルを差し替えても古いキャッシュが残る（shape 不一致で
      // ORT_INVALID_ARGUMENT になる）。起動時に必ず消して最新を展開させる。
      try {
        final dir = await getTemporaryDirectory();
        for (final name in ['rtmdet_ins.onnx', 'attr_cls.onnx']) {
          final cached = File('${dir.path}${Platform.pathSeparator}$name');
          if (await cached.exists()) {
            await cached.delete();
          }
        }
      } catch (_) {}
      await _detector.init();
      // 属性分類器はオプショナル: 失敗しても検出は動かす
      try {
        await _attrCls.init();
      } catch (e) {
        debugPrint('attr classifier unavailable: $e');
      }
    } catch (e) {
      _setStatus('Model load failed: $e');
      return;
    }

    try {
      _setStatus('Opening camera…');
      final cameras = await availableCameras();
      final back = cameras.firstWhere(
        (c) => c.lensDirection == CameraLensDirection.back,
        orElse: () => cameras.first,
      );
      final controller = CameraController(
        back,
        ResolutionPreset.medium,
        enableAudio: false,
        imageFormatGroup: ImageFormatGroup.yuv420,
      );
      await controller.initialize();
      if (_disposed) {
        await controller.dispose();
        return;
      }
      _controller = controller;
      await controller.startImageStream(_onFrame);
      _setStatus('Running');
      setState(() {});
    } catch (e) {
      _setStatus('Camera error: $e');
    }
  }

  Future<void> _onFrame(CameraImage image) async {
    if (_running || _disposed) return;
    if (DateTime.now().difference(_lastStart) < _minInterval) return;
    _running = true;
    _lastStart = DateTime.now();
    try {
      final started = DateTime.now();
      final sw = Stopwatch()..start();
      final rotation = _controller?.description.sensorOrientation ?? 90;
      final rgba = _yuv420ToRgba(image, _detector.inputSize, rotation);
      final yuvMs = sw.elapsedMilliseconds;
      final bool swap = rotation == 90 || rotation == 270;
      final double aspect = swap
          ? image.height / image.width
          : image.width / image.height;
      final res = await _detector.runRaw(rgba);
      final overlay = _detector.composeMasks(res);
      sw.reset();
      final img = await _decodeMask(overlay, _detector.inputSize);
      final decMs = sw.elapsedMilliseconds;
      final ms = DateTime.now().difference(started).inMilliseconds;
      final st = _detector.lastStageMs;
      final timing = 'yuv $yuvMs · ten ${st['ten']} · run ${st['run']}'
          ' · dets ${st['dets']} · masks ${st['masks']}'
          ' · ovl ${st['ovl']} · dec $decMs';

      // タップ追跡の更新と、トラックごとの部位（キャップ/ラベル）切り抜き
      for (final t in _tracker.update(res.detections)) {
        t.disposeImages(); // ロストで破棄されたトラック
      }
      final ids = Map<Detection, int>.identity();
      final capOf =
          MultiTracker.assignParts(_tracker.tracks, res.detections, 1);
      final lblOf =
          MultiTracker.assignParts(_tracker.tracks, res.detections, 2);
      for (final t in _tracker.tracks) {
        if (t.lastMatch != null) ids[t.lastMatch!] = t.id;
        final cap = capOf[t];
        final lbl = lblOf[t];
        if (cap != null) {
          final im = await _decodeCrop(rgba, cap.rect);
          if (im != null) {
            t.capImg?.dispose();
            t.capImg = im;
          }
        }
        if (lbl != null) {
          final im = await _decodeCrop(rgba, lbl.rect);
          if (im != null) {
            t.labelImg?.dispose();
            t.labelImg = im;
          }
        }
      }

      // 属性推定は検出と直列に低頻度で回す（検出との CPU スレッド競合を
      // 避ける。_running ガード内なので次フレームとも重ならない）
      if (_attrCls.isReady && _tracker.active && _attrFails < _maxAttrFails) {
        await _updateTrackAttrs(image, rotation, res.detections);
      }

      if (!_disposed) {
        _mask?.dispose();
        final c = _detector.lastCounts;
        // FPS はマスク更新間隔の EMA（推論+変換+スロットル込みの実効値）
        final now = DateTime.now();
        final gapMs = now.difference(_lastDone).inMilliseconds;
        _lastDone = now;
        if (gapMs > 0 && gapMs < 10000) {
          final inst = 1000.0 / gapMs;
          _fps = _fps == 0 ? inst : _fps * 0.8 + inst * 0.2;
        }
        setState(() {
          _mask = img;
          _frameAspect = aspect;
          _lastInferMs = ms;
          _timing = timing;
          _lastDets = res.detections;
          _boxes = [
            for (final d in res.detections)
              (
                rect: d.rect,
                cls: d.cls,
                trackId: ids[d],
              ),
          ];
          _status = 'bottle:${c[0]} cap:${c[1]} label:${c[2]}';
        });
      } else {
        img.dispose();
      }
    } catch (e) {
      // フレーム破棄でループ継続（原因はステータスに表示して可視化）
      _setStatus('ERR: ${e.toString().substring(0, e.toString().length.clamp(0, 120))}');
    } finally {
      _running = false;
    }
  }

  /// YUV420 の CameraImage を、センサー回転を補正した**フレーム全体**の
  /// RGBA (s×s、アスペクトは squash) に変換する。ニアレストネイバー。
  Uint8List _yuv420ToRgba(CameraImage image, int s, int rotation) {
    final int w = image.width;
    final int h = image.height;
    final yPlane = image.planes[0];
    final uPlane = image.planes[1];
    final vPlane = image.planes[2];
    final int yStride = yPlane.bytesPerRow;
    final int uvStride = uPlane.bytesPerRow;
    final int uvPixStride = uPlane.bytesPerPixel ?? 1;

    final bool swap = rotation == 90 || rotation == 270;
    final int rw = swap ? h : w;
    final int rh = swap ? w : h;

    final out = Uint8List(s * s * 4);
    for (int dy = 0; dy < s; dy++) {
      final int ry = dy * rh ~/ s;
      for (int dx = 0; dx < s; dx++) {
        final int rx = dx * rw ~/ s;
        int sx, sy;
        switch (rotation) {
          case 90:
            sx = ry;
            sy = h - 1 - rx;
            break;
          case 180:
            sx = w - 1 - rx;
            sy = h - 1 - ry;
            break;
          case 270:
            sx = w - 1 - ry;
            sy = rx;
            break;
          default:
            sx = rx;
            sy = ry;
        }

        final int yv = yPlane.bytes[sy * yStride + sx];
        final int uvIndex = (sy >> 1) * uvStride + (sx >> 1) * uvPixStride;
        final int u = uPlane.bytes[uvIndex] - 128;
        final int v = vPlane.bytes[uvIndex] - 128;

        int r = (yv + 1.402 * v).round();
        int g = (yv - 0.344136 * u - 0.714136 * v).round();
        int b = (yv + 1.772 * u).round();
        r = r.clamp(0, 255);
        g = g.clamp(0, 255);
        b = b.clamp(0, 255);

        final int o = (dy * s + dx) * 4;
        out[o] = r;
        out[o + 1] = g;
        out[o + 2] = b;
        out[o + 3] = 255;
      }
    }
    return out;
  }

  Future<ui.Image> _decodeMask(Uint8List rgba, int size) {
    final completer = Completer<ui.Image>();
    ui.decodeImageFromPixels(
      rgba,
      size,
      size,
      ui.PixelFormat.rgba8888,
      completer.complete,
    );
    return completer.future;
  }

  /// トラック中ボトルの属性を推定して EMA 集約に足し込む。
  ///
  /// クロップは 320 入力ではなくカメラの元解像度 YUV から直接切り出す
  /// （320 入力上のボトルは学習条件の長辺>=96px を満たせないことが多い）。
  /// 検出時のみ・トラックごとに [_attrInterval] のレート制限つき。
  Future<void> _updateTrackAttrs(
      CameraImage image, int rotation, List<Detection> dets) async {
    try {
      final bool swap = rotation == 90 || rotation == 270;
      final int rw = swap ? image.height : image.width; // 回転補正後の幅/高さ
      final int rh = swap ? image.width : image.height;
      final double s = _detector.inputSize.toDouble();
      final now = DateTime.now();
      var updated = false;
      final capOf = MultiTracker.assignParts(_tracker.tracks, dets, 1);
      final lblOf = MultiTracker.assignParts(_tracker.tracks, dets, 2);
      for (final t in List<Track>.of(_tracker.tracks)) {
        if (t.lastMatch == null) continue; // 見失い中の位置では推論しない
        if (now.difference(t.lastAttrAt) < _attrInterval) continue;
        // 学習時と同じ余白（cropRgba と同係数）を入力座標系で付ける
        final double pad = t.rect.shortestSide * 0.15 + 2;
        final r = ui.Rect.fromLTRB(
          (t.rect.left - pad).clamp(0.0, s),
          (t.rect.top - pad).clamp(0.0, s),
          (t.rect.right + pad).clamp(0.0, s),
          (t.rect.bottom + pad).clamp(0.0, s),
        );
        // 入力座標（フレーム全体の squash）→ 回転補正後フレーム座標
        final crop = ui.Rect.fromLTRB(
          r.left * rw / s,
          r.top * rh / s,
          r.right * rw / s,
          r.bottom * rh / s,
        );
        // 学習データの付与条件（元解像度で長辺 >= 96px）未満はスキップ
        if (crop.width < 1 ||
            crop.height < 1 ||
            (crop.width < AttrClassifier.minCropSide &&
                crop.height < AttrClassifier.minCropSide)) {
          continue;
        }
        final rgba =
            _yuvCropToRgba(image, rotation, crop, AttrClassifier.inputSize);
        try {
          t.attrs.add(await _attrCls.run(rgba));
          // 検出器の cap/label パーツ有無はより強い証拠なので融合する。
          // 「あり」は誤検出が少なく強め、「なし」は角度・遮蔽で見えない
          // だけの可能性があるため弱めに効かせる。
          final capDet = capOf[t];
          final lblDet = lblOf[t];
          t.attrs.addObservation('cap', capDet != null ? 0 : 1,
              alpha: capDet != null ? 0.35 : 0.15);
          t.attrs.addObservation('label', lblDet != null ? 0 : 1,
              alpha: lblDet != null ? 0.35 : 0.15);
          t.lastAttrAt = DateTime.now();
          _attrFails = 0;
          updated = true;
        } catch (e) {
          _attrFails++;
          debugPrint('attr inference failed ($_attrFails/$_maxAttrFails): $e');
          return; // 次の検出サイクルで再試行（連続失敗でヒューズが切れる）
        }
      }
      if (updated && !_disposed) setState(() {});
    } catch (e) {
      debugPrint('attr update failed: $e');
    }
  }

  /// YUV420 フレームから、回転補正後座標の [crop] 領域を [s]×[s] の RGBA に
  /// 切り出す（アスペクトは潰す=学習時と同じ規約。ニアレストネイバー）。
  Uint8List _yuvCropToRgba(
      CameraImage image, int rotation, ui.Rect crop, int s) {
    final int w = image.width;
    final int h = image.height;
    final yPlane = image.planes[0];
    final uPlane = image.planes[1];
    final vPlane = image.planes[2];
    final int yStride = yPlane.bytesPerRow;
    final int uvStride = uPlane.bytesPerRow;
    final int uvPixStride = uPlane.bytesPerPixel ?? 1;

    final bool swap = rotation == 90 || rotation == 270;
    final int rw = swap ? h : w;
    final int rh = swap ? w : h;

    final out = Uint8List(s * s * 4);
    for (int dy = 0; dy < s; dy++) {
      final int ry =
          (crop.top + (dy + 0.5) * crop.height / s).floor().clamp(0, rh - 1);
      for (int dx = 0; dx < s; dx++) {
        final int rx =
            (crop.left + (dx + 0.5) * crop.width / s).floor().clamp(0, rw - 1);
        int sx, sy;
        switch (rotation) {
          case 90:
            sx = ry;
            sy = h - 1 - rx;
            break;
          case 180:
            sx = w - 1 - rx;
            sy = h - 1 - ry;
            break;
          case 270:
            sx = w - 1 - ry;
            sy = rx;
            break;
          default:
            sx = rx;
            sy = ry;
        }

        final int yv = yPlane.bytes[sy * yStride + sx];
        final int uvIndex = (sy >> 1) * uvStride + (sx >> 1) * uvPixStride;
        final int u = uPlane.bytes[uvIndex] - 128;
        final int v = vPlane.bytes[uvIndex] - 128;

        int r = (yv + 1.402 * v).round();
        int g = (yv - 0.344136 * u - 0.714136 * v).round();
        int b = (yv + 1.772 * u).round();
        r = r.clamp(0, 255);
        g = g.clamp(0, 255);
        b = b.clamp(0, 255);

        final int o = (dy * s + dx) * 4;
        out[o] = r;
        out[o + 1] = g;
        out[o + 2] = b;
        out[o + 3] = 255;
      }
    }
    return out;
  }

  /// タップ: 追跡中ボトル→そのトラックを解除 / 未追跡ボトル→トラック追加 /
  /// ボトル以外→全トラック解除。
  void _onTapDown(Offset local, Size size) {
    final p = MaskPainter.screenToInput(
        local, size, _frameAspect, _detector.inputSize);
    setState(() {
      final hit = _tracker.trackAt(p);
      if (hit != null) {
        _tracker.remove(hit);
        hit.disposeImages();
      } else if (!_tracker.addAt(p, _lastDets)) {
        for (final t in _tracker.clear()) {
          t.disposeImages();
        }
      }
    });
  }

  /// [rgba]（入力フレーム）から [rect] を切り抜いて ui.Image 化する。
  Future<ui.Image?> _decodeCrop(Uint8List rgba, ui.Rect rect) async {
    final c = cropRgba(rgba, _detector.inputSize, rect);
    if (c == null) return null;
    final completer = Completer<ui.Image>();
    ui.decodeImageFromPixels(
        c.rgba, c.width, c.height, ui.PixelFormat.rgba8888, completer.complete);
    return completer.future;
  }

  void _setStatus(String s) {
    if (_disposed) return;
    setState(() => _status = s);
  }

  @override
  void dispose() {
    _disposed = true;
    _mask?.dispose();
    for (final t in _tracker.clear()) {
      t.disposeImages();
    }
    _controller?.dispose();
    _detector.dispose();
    _attrCls.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final controller = _controller;
    return Stack(
      fit: StackFit.expand,
      children: [
        const ColoredBox(color: Colors.black),
        if (controller != null && controller.value.isInitialized)
          FittedBox(
            fit: BoxFit.cover,
            clipBehavior: Clip.hardEdge,
            child: SizedBox(
              width: controller.value.previewSize!.height,
              height: controller.value.previewSize!.width,
              child: CameraPreview(controller),
            ),
          ),
        if (_mask != null)
          Positioned.fill(
            child: CustomPaint(
              painter: MaskPainter(_mask!,
                  srcAspect: _frameAspect, boxes: _boxes, palette: _palette),
            ),
          ),
        // タップでボトルを追跡選択
        Positioned.fill(
          child: LayoutBuilder(
            builder: (context, c) => GestureDetector(
              behavior: HitTestBehavior.translucent,
              onTapDown: (d) => _onTapDown(d.localPosition, c.biggest),
            ),
          ),
        ),
        if (_tracker.active)
          Positioned(
            left: 0,
            right: 0,
            bottom: 24,
            child: SingleChildScrollView(
              scrollDirection: Axis.horizontal,
              padding: const EdgeInsets.symmetric(horizontal: 12),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  for (final t in _tracker.tracks) ...[
                    TrackPanel(
                      trackId: t.id,
                      cap: t.capImg,
                      label: t.labelImg,
                      capColor: _palette[1],
                      labelColor: _palette[2],
                      attrs: t.attrs.display(),
                    ),
                    const SizedBox(width: 10),
                  ],
                ],
              ),
            ),
          ),
        Positioned(
          left: 12,
          top: 12,
          child: StatusChip(
              status: _status,
              inferMs: _lastInferMs,
              fps: _fps,
              detail: _timing),
        ),
      ],
    );
  }
}
