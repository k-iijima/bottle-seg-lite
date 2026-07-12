import 'dart:async';
import 'dart:js_interop';
import 'dart:typed_data';
import 'dart:ui' as ui;
import 'dart:ui_web' as ui_web;

import 'package:flutter/material.dart';
import 'package:web/web.dart' as web;

import 'detector.dart';
import 'overlay_paint.dart';

/// Web 向け: ネイティブ <video> のプレビューに RTMDet-Ins の検出オーバーレイを
/// 重ねる（onnxruntime-web / wasm）。プレビューはプラットフォームビューなので
/// 推論と独立して滑らかに再生され、推論はフレームドロップ式。
class CameraSegView extends StatefulWidget {
  const CameraSegView({super.key});

  @override
  State<CameraSegView> createState() => _CameraSegViewState();
}

class _CameraSegViewState extends State<CameraSegView> {
  static const String _viewType = 'camera-video-view';

  final Detector _detector = Detector(inputSize: 416);
  double _frameAspect = 1.0;

  late final web.HTMLVideoElement _video;
  late final web.HTMLCanvasElement _grabCanvas;
  late final web.CanvasRenderingContext2D _grabCtx;
  web.MediaStream? _stream;

  ui.Image? _mask;
  bool _running = false;
  bool _disposed = false;

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
    final s = _detector.inputSize;
    _grabCanvas = web.HTMLCanvasElement()
      ..width = s
      ..height = s;
    _grabCtx = _grabCanvas.getContext('2d') as web.CanvasRenderingContext2D;

    _video = web.HTMLVideoElement()
      ..autoplay = true
      ..muted = true
      ..setAttribute('playsinline', '');
    _video.style.setProperty('width', '100%');
    _video.style.setProperty('height', '100%');
    _video.style.setProperty('object-fit', 'cover');

    ui_web.platformViewRegistry
        .registerViewFactory(_viewType, (int _) => _video);

    try {
      _setStatus('Requesting camera…');
      final constraints = web.MediaStreamConstraints(video: true.toJS);
      final stream = await web.window.navigator.mediaDevices
          .getUserMedia(constraints)
          .toDart;
      _stream = stream;
      _video.srcObject = stream;
      await _video.play().toDart;
    } catch (e) {
      _setStatus('Camera error: $e');
      return;
    }

    try {
      _setStatus('Loading model…');
      await _detector.init();
    } catch (e) {
      _setStatus('Model load failed (did you export rtmdet_ins.onnx?): $e');
      return;
    }

    _setStatus('Running');
    unawaited(_loop());
  }

  Future<void> _loop() async {
    while (!_disposed) {
      if (!_running && _video.readyState >= 2 && _video.videoWidth > 0) {
        _running = true;
        try {
          final started = DateTime.now();
          final rgba = _grabFrame();
          final overlay = await _detector.run(rgba);
          final img = await _decodeMask(overlay, _detector.inputSize);
          final ms = DateTime.now().difference(started).inMilliseconds;
          if (!_disposed) {
            _mask?.dispose();
            final c = _detector.lastCounts;
            final now = DateTime.now();
            final gapMs = now.difference(_lastDone).inMilliseconds;
            _lastDone = now;
            if (gapMs > 0 && gapMs < 10000) {
              final inst = 1000.0 / gapMs;
              _fps = _fps == 0 ? inst : _fps * 0.8 + inst * 0.2;
            }
            setState(() {
              _mask = img;
              _frameAspect = _video.videoWidth > 0
                  ? _video.videoWidth / _video.videoHeight
                  : 1.0;
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
      await Future<void>.delayed(const Duration(milliseconds: 1));
    }
  }

  Uint8List _grabFrame() {
    final s = _detector.inputSize;
    _grabCtx.drawImage(_video, 0, 0, s.toDouble(), s.toDouble());
    final imageData = _grabCtx.getImageData(0, 0, s, s);
    final clamped = imageData.data.toDart;
    return clamped.buffer
        .asUint8List(clamped.offsetInBytes, clamped.lengthInBytes);
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
    final stream = _stream;
    if (stream != null) {
      final tracks = stream.getTracks().toDart;
      for (final t in tracks) {
        t.stop();
      }
    }
    _detector.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Stack(
      fit: StackFit.expand,
      children: [
        const ColoredBox(color: Colors.black),
        const HtmlElementView(viewType: _viewType),
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
