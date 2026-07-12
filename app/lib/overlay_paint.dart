import 'dart:ui' as ui;

import 'package:flutter/material.dart';

/// 検出オーバーレイ画像を、カメラプレビュー(object-fit: cover)と同じ変換で
/// 画面に重ねる Painter（web / mobile 共用）。
///
/// [srcAspect] はオーバーレイが表す**元フレームの縦横比**（幅/高さ）。
/// オーバーレイ画像自体は正方（モデル入力にフレーム全体を squash したもの）でも、
/// 論理的にはフレーム全体を表しているため、論理アスペクトで cover クロップして
/// からピクセル座標に写像することでプレビューと正確に一致する。
class MaskPainter extends CustomPainter {
  MaskPainter(this.mask, {this.srcAspect = 1.0});

  final ui.Image mask;
  final double srcAspect;

  @override
  void paint(Canvas canvas, Size size) {
    final dst = Offset.zero & size;
    // 論理フレーム（アスペクトのみ意味を持つ）で cover クロップ
    final logical = Rect.fromLTWH(0, 0, srcAspect, 1);
    final crop = coverSrcRect(logical, dst);
    // 論理座標 → マスク画像のピクセル座標
    final sx = mask.width / srcAspect;
    final sy = mask.height.toDouble();
    final srcPx = Rect.fromLTWH(
        crop.left * sx, crop.top * sy, crop.width * sx, crop.height * sy);
    canvas.drawImageRect(
      mask,
      srcPx,
      dst,
      Paint()..filterQuality = FilterQuality.low,
    );
  }

  /// src を dst のアスペクト比に合わせて中央クロップする。
  static Rect coverSrcRect(Rect src, Rect dst) {
    final srcAspect = src.width / src.height;
    final dstAspect = dst.width / dst.height;
    if ((srcAspect - dstAspect).abs() < 1e-3) return src;
    if (dstAspect > srcAspect) {
      final h = src.width / dstAspect;
      final dy = (src.height - h) / 2;
      return Rect.fromLTWH(src.left, src.top + dy, src.width, h);
    } else {
      final w = src.height * dstAspect;
      final dx = (src.width - w) / 2;
      return Rect.fromLTWH(src.left + dx, src.top, w, src.height);
    }
  }

  @override
  bool shouldRepaint(MaskPainter oldDelegate) =>
      oldDelegate.mask != mask || oldDelegate.srcAspect != srcAspect;
}

class StatusChip extends StatelessWidget {
  const StatusChip(
      {super.key, required this.status, this.inferMs = 0, this.fps = 0});

  final String status;
  final int inferMs;
  final double fps;

  @override
  Widget build(BuildContext context) {
    final parts = [
      status,
      if (inferMs > 0) '${inferMs}ms',
      if (fps > 0) '${fps.toStringAsFixed(1)}fps',
    ];
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
      decoration: BoxDecoration(
        color: Colors.black54,
        borderRadius: BorderRadius.circular(8),
      ),
      child: Text(
        parts.join('  •  '),
        style: const TextStyle(color: Colors.white, fontSize: 12),
      ),
    );
  }
}
