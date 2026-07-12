import 'dart:async';
import 'dart:io';
import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';

import 'detector.dart';
import 'overlay_paint.dart';

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
        final cached =
            File('${dir.path}${Platform.pathSeparator}rtmdet_ins.onnx');
        if (await cached.exists()) {
          await cached.delete();
        }
      } catch (_) {}
      await _detector.init();
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
      final rotation = _controller?.description.sensorOrientation ?? 90;
      final rgba = _yuv420ToRgba(image, _detector.inputSize, rotation);
      final bool swap = rotation == 90 || rotation == 270;
      final double aspect = swap
          ? image.height / image.width
          : image.width / image.height;
      final overlay = await _detector.run(rgba);
      final img = await _decodeMask(overlay, _detector.inputSize);
      final ms = DateTime.now().difference(started).inMilliseconds;
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

  void _setStatus(String s) {
    if (_disposed) return;
    setState(() => _status = s);
  }

  @override
  void dispose() {
    _disposed = true;
    _mask?.dispose();
    _controller?.dispose();
    _detector.dispose();
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
              painter: MaskPainter(_mask!, srcAspect: _frameAspect),
            ),
          ),
        Positioned(
          left: 12,
          top: 12,
          child: StatusChip(status: _status, inferMs: _lastInferMs, fps: _fps),
        ),
      ],
    );
  }
}
