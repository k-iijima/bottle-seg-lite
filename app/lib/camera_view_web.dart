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

  /// 選択可能なカメラ一覧（許可取得後に enumerateDevices で取得）。
  List<({String id, String label})> _cameras = const [];
  String? _currentCameraId;

  /// 前回選んだカメラを次回も使うための localStorage キー。
  /// ブラウザ既定が仮想カメラ等で黒映像になる環境への対策。
  static const String _cameraPrefKey = 'camera_device_id';

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
      final saved = web.window.localStorage.getItem(_cameraPrefKey);
      web.MediaStream stream;
      try {
        stream = await _openStream(saved);
      } catch (_) {
        // 保存していたデバイスが外された等 → 既定カメラでやり直す
        if (saved == null) rethrow;
        web.window.localStorage.removeItem(_cameraPrefKey);
        stream = await _openStream(null);
      }
      await _attachStream(stream);
      await _refreshCameraList();
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

  /// deviceId 指定（null なら既定カメラ）で MediaStream を開く。
  Future<web.MediaStream> _openStream(String? deviceId) {
    final constraints = web.MediaStreamConstraints(
      video: deviceId == null
          ? true.toJS
          : {'deviceId': {'exact': deviceId}}.jsify()!,
    );
    return web.window.navigator.mediaDevices.getUserMedia(constraints).toDart;
  }

  Future<void> _attachStream(web.MediaStream stream) async {
    _stream = stream;
    _video.srcObject = stream;
    await _video.play().toDart;
    final tracks = stream.getVideoTracks().toDart;
    if (tracks.isNotEmpty) {
      _currentCameraId = tracks.first.getSettings().deviceId;
    }
  }

  void _stopStream() {
    final stream = _stream;
    if (stream == null) return;
    for (final t in stream.getTracks().toDart) {
      t.stop();
    }
    _stream = null;
  }

  /// ビデオ入力デバイスを列挙する（ラベルは許可取得後でないと空になる）。
  Future<void> _refreshCameraList() async {
    final devices = (await web.window.navigator.mediaDevices
            .enumerateDevices()
            .toDart)
        .toDart;
    final cams = <({String id, String label})>[];
    for (final d in devices) {
      if (d.kind == 'videoinput') {
        cams.add((
          id: d.deviceId,
          label: d.label.isEmpty ? 'Camera ${cams.length + 1}' : d.label,
        ));
      }
    }
    if (!_disposed) setState(() => _cameras = cams);
  }

  Future<void> _switchCamera(String deviceId) async {
    if (deviceId == _currentCameraId) return;
    final prev = _currentCameraId;
    _setStatus('Switching camera…');
    // モバイル等ではカメラの同時オープンに失敗するため、先に止めてから開く
    _stopStream();
    try {
      await _attachStream(await _openStream(deviceId));
      web.window.localStorage.setItem(_cameraPrefKey, deviceId);
      _setStatus('Camera switched');
    } catch (e) {
      _setStatus('Camera switch failed: $e');
      try {
        await _attachStream(await _openStream(prev));
      } catch (_) {}
    }
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
    _stopStream();
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
        if (_cameras.isNotEmpty)
          Positioned(
            right: 12,
            top: 12,
            child: PopupMenuButton<String>(
              tooltip: 'カメラ切替',
              onSelected: _switchCamera,
              itemBuilder: (_) => [
                for (final c in _cameras)
                  CheckedPopupMenuItem<String>(
                    value: c.id,
                    checked: c.id == _currentCameraId,
                    child: Text(c.label, overflow: TextOverflow.ellipsis),
                  ),
              ],
              child: Container(
                padding: const EdgeInsets.all(6),
                decoration: BoxDecoration(
                  color: Colors.black54,
                  borderRadius: BorderRadius.circular(8),
                ),
                child: const Icon(Icons.cameraswitch,
                    color: Colors.white, size: 20),
              ),
            ),
          ),
      ],
    );
  }
}
