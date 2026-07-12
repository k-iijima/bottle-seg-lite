import 'dart:async';
import 'dart:js_interop';
import 'dart:js_interop_unsafe';
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

  final Detector _detector = Detector(inputSize: 320);
  double _frameAspect = 1.0;

  late final web.HTMLVideoElement _video;
  late final web.HTMLCanvasElement _grabCanvas;
  late final web.CanvasRenderingContext2D _grabCtx;
  web.MediaStream? _stream;

  /// 選択可能なカメラ一覧（許可取得後に enumerateDevices で取得）。
  List<({String id, String label})> _cameras = const [];
  String? _currentCameraId;

  /// モデル×実行方式の切替候補（GPU=WebGPU 優先・非対応時は wasm に自動フォールバック）。
  /// int8×GPU は ort-web の WebGPU が per-channel DequantizeLinear 未対応で
  /// 推論実行時にカーネルエラーになるため提供しない（int8 は CPU 専用）。
  static const List<({String label, String asset, bool gpu})> _inferConfigs = [
    (label: 'fp32 / GPU', asset: Detector.fp32Asset, gpu: true),
    (label: 'fp32 / CPU', asset: Detector.fp32Asset, gpu: false),
    (label: 'int8 / CPU', asset: Detector.int8Asset, gpu: false),
  ];
  int _configIndex = 0;

  /// WebGPU アダプタが実際に取得できたか（指定してもフォールバックされうるため、
  /// requestAdapter で確認した結果を表示に使う）。
  bool _webGpuOk = false;

  /// 実際に動いているモードの表示用ラベル（例: fp32/GPU, int8/CPU×8）。
  String get _modeLabel {
    final c = _inferConfigs[_configIndex];
    final model = c.asset == Detector.int8Asset ? 'int8' : 'fp32';
    if (c.gpu && _webGpuOk) return '$model/GPU';
    final t = _wasmThreads;
    final cpu = 'CPU${t > 1 ? '×$t' : ''}';
    return c.gpu ? '$model/GPU→$cpu' : '$model/$cpu';
  }

  /// ort-web の wasm スレッド数（crossOriginIsolated でなければ 1）。
  int get _wasmThreads {
    if (!web.window.crossOriginIsolated) return 1;
    final ort = (web.window as JSObject).getProperty('ort'.toJS) as JSObject?;
    final env = ort?.getProperty('env'.toJS) as JSObject?;
    final wasm = env?.getProperty('wasm'.toJS) as JSObject?;
    final n = wasm?.getProperty('numThreads'.toJS);
    return n.isA<JSNumber>() ? (n! as JSNumber).toDartInt : 1;
  }

  Future<void> _probeWebGpu() async {
    try {
      final gpu =
          (web.window.navigator as JSObject).getProperty('gpu'.toJS) as JSObject?;
      if (gpu == null) return;
      final adapter = await (gpu.callMethod('requestAdapter'.toJS)
              as JSPromise<JSAny?>)
          .toDart;
      _webGpuOk = adapter != null;
    } catch (_) {
      _webGpuOk = false;
    }
  }

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

  /// ステージ別所要時間の表示文字列（ボトルネック特定用）。
  String _timing = '';

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
    // getImageData を毎フレーム呼ぶため readback 最適化を指定
    _grabCtx = _grabCanvas.getContext(
      '2d',
      {'willReadFrequently': true}.jsify(),
    ) as web.CanvasRenderingContext2D;

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

    await _probeWebGpu();

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

  Future<void> _switchConfig(int index) async {
    if (index == _configIndex) return;
    final prev = _configIndex;
    final c = _inferConfigs[index];
    setState(() => _configIndex = index);
    _setStatus('Loading model… (${c.label})');
    // 実行中の推論が終わるのを待ってからセッションを差し替える
    while (_running) {
      await Future<void>.delayed(const Duration(milliseconds: 20));
    }
    try {
      await _detector.load(modelAsset: c.asset, preferGpu: c.gpu);
      _setStatus('Running (${c.label})');
    } catch (e) {
      final p = _inferConfigs[prev];
      setState(() => _configIndex = prev);
      _setStatus('Switch failed → ${p.label}: '
          '${e.toString().substring(0, e.toString().length.clamp(0, 80))}');
      try {
        await _detector.load(modelAsset: p.asset, preferGpu: p.gpu);
      } catch (_) {}
    }
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
      if (!_running &&
          _detector.isReady &&
          _video.readyState >= 2 &&
          _video.videoWidth > 0) {
        _running = true;
        try {
          final started = DateTime.now();
          final sw = Stopwatch()..start();
          final rgba = _grabFrame();
          final grabMs = sw.elapsedMilliseconds;
          final overlay = await _detector.run(rgba);
          sw.reset();
          final img = await _decodeMask(overlay, _detector.inputSize);
          final decMs = sw.elapsedMilliseconds;
          final ms = DateTime.now().difference(started).inMilliseconds;
          final st = _detector.lastStageMs;
          final timing = 'grab $grabMs · ten ${st['ten']}'
              ' · run ${st['run']} · dets ${st['dets']}'
              ' · masks ${st['masks']} · ovl ${st['ovl']} · dec $decMs';
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
              _timing = timing;
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
          child: StatusChip(
              mode: _modeLabel,
              status: _status,
              inferMs: _lastInferMs,
              fps: _fps,
              detail: _timing),
        ),
        Positioned(
          right: 12,
          top: 12,
          child: Row(
            children: [
              PopupMenuButton<int>(
                tooltip: 'モデル/実行方式',
                onSelected: _switchConfig,
                itemBuilder: (_) => [
                  for (var i = 0; i < _inferConfigs.length; i++)
                    CheckedPopupMenuItem<int>(
                      value: i,
                      checked: i == _configIndex,
                      child: Text(_inferConfigs[i].label),
                    ),
                ],
                child: _chipButton(Icons.tune),
              ),
              if (_cameras.isNotEmpty) ...[
                const SizedBox(width: 8),
                PopupMenuButton<String>(
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
                  child: _chipButton(Icons.cameraswitch),
                ),
              ],
            ],
          ),
        ),
      ],
    );
  }

  Widget _chipButton(IconData icon) {
    return Container(
      padding: const EdgeInsets.all(6),
      decoration: BoxDecoration(
        color: Colors.black54,
        borderRadius: BorderRadius.circular(8),
      ),
      child: Icon(icon, color: Colors.white, size: 20),
    );
  }
}
