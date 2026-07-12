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
  MaskPainter(this.mask, {this.srcAspect = 1.0, this.boxes = const [],
      this.palette = const []});

  final ui.Image mask;
  final double srcAspect;

  /// マスク画像と同じピクセル座標系のボックス（Web: 検出間は外挿で更新される）。
  /// モバイルはオーバーレイに焼き込むため空。
  final List<({Rect rect, int cls})> boxes;
  final List<Color> palette;

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

    if (boxes.isEmpty) return;
    // マスクと同じ srcPx→dst 変換でボックスを描く
    final double kx = dst.width / srcPx.width;
    final double ky = dst.height / srcPx.height;
    final stroke = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2;
    for (final b in boxes) {
      stroke.color = b.cls < palette.length ? palette[b.cls] : Colors.white;
      canvas.drawRect(
        Rect.fromLTRB(
          (b.rect.left - srcPx.left) * kx,
          (b.rect.top - srcPx.top) * ky,
          (b.rect.right - srcPx.left) * kx,
          (b.rect.bottom - srcPx.top) * ky,
        ),
        stroke,
      );
    }
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
      oldDelegate.mask != mask ||
      oldDelegate.srcAspect != srcAspect ||
      oldDelegate.boxes != boxes;
}

class StatusChip extends StatelessWidget {
  const StatusChip(
      {super.key,
      required this.status,
      this.mode,
      this.inferMs = 0,
      this.fps = 0,
      this.detail});

  final String status;

  /// 実行中のモード表示（例: fp32/GPU）。null なら省略（mobile など）。
  final String? mode;
  final int inferMs;
  final double fps;

  /// ステージ別タイミング等の補足行。null / 空なら省略。
  final String? detail;

  @override
  Widget build(BuildContext context) {
    final parts = [
      if (mode != null) mode!,
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
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisSize: MainAxisSize.min,
        children: [
          Text(
            parts.join('  •  '),
            style: const TextStyle(color: Colors.white, fontSize: 12),
          ),
          if (detail != null && detail!.isNotEmpty)
            Text(
              detail!,
              style: const TextStyle(color: Colors.white70, fontSize: 10),
            ),
        ],
      ),
    );
  }
}
