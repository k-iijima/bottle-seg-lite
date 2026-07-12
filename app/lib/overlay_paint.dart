import 'dart:ui' as ui;

import 'package:flutter/material.dart';

import 'attr_classifier.dart';

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
  /// trackId 付きのボックスは白の太枠+ID チップでハイライトする。
  final List<({Rect rect, int cls, int? trackId})> boxes;
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
    final stroke = Paint()..style = PaintingStyle.stroke;
    for (final b in boxes) {
      final r = Rect.fromLTRB(
        (b.rect.left - srcPx.left) * kx,
        (b.rect.top - srcPx.top) * ky,
        (b.rect.right - srcPx.left) * kx,
        (b.rect.bottom - srcPx.top) * ky,
      );
      final color = b.cls < palette.length ? palette[b.cls] : Colors.white;
      if (b.trackId != null) {
        stroke
          ..color = Colors.white
          ..strokeWidth = 5;
        canvas.drawRect(r, stroke);
      }
      stroke
        ..color = color
        ..strokeWidth = b.trackId != null ? 2.5 : 2;
      canvas.drawRect(r, stroke);

      if (b.trackId != null) {
        // トラック ID チップ（枠の左上。画面上端では枠内に落とす）
        final tp = TextPainter(
          text: TextSpan(
            text: '#${b.trackId}',
            style: const TextStyle(
                color: Colors.white, fontSize: 13, fontWeight: FontWeight.bold),
          ),
          textDirection: TextDirection.ltr,
        )..layout();
        const double padX = 5;
        final double chipH = tp.height + 4;
        final double top = r.top >= chipH ? r.top - chipH : r.top;
        final chip = Rect.fromLTWH(r.left, top, tp.width + padX * 2, chipH);
        canvas.drawRect(chip, Paint()..color = color);
        tp.paint(canvas, Offset(chip.left + padX, chip.top + 2));
      }
    }
  }

  /// 画面座標 [p] を、paint と同じ cover 変換の逆写像で
  /// オーバーレイ入力ピクセル座標に変換する（タップ選択用）。
  static Offset screenToInput(
      Offset p, Size size, double srcAspect, int inputSize) {
    final dst = Offset.zero & size;
    final logical = Rect.fromLTWH(0, 0, srcAspect, 1);
    final crop = coverSrcRect(logical, dst);
    final sx = inputSize / srcAspect;
    final sy = inputSize.toDouble();
    final srcPx = Rect.fromLTWH(
        crop.left * sx, crop.top * sy, crop.width * sx, crop.height * sy);
    return Offset(
      srcPx.left + p.dx / dst.width * srcPx.width,
      srcPx.top + p.dy / dst.height * srcPx.height,
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
      oldDelegate.mask != mask ||
      oldDelegate.srcAspect != srcAspect ||
      oldDelegate.boxes != boxes;
}

/// 追跡中トラック 1 件分のパネル（#ID ヘッダ+cap/label の切り抜きタイル）。
class TrackPanel extends StatelessWidget {
  const TrackPanel({
    super.key,
    required this.trackId,
    required this.cap,
    required this.label,
    required this.capColor,
    required this.labelColor,
    this.attrs = const {},
  });

  final int trackId;

  /// 切り抜き画像（未検出時は null で「—」表示）。
  final ui.Image? cap;
  final ui.Image? label;
  final Color capColor;
  final Color labelColor;

  /// 属性分類器のトラック内集約結果（AttrAggregate.display()）。
  /// 空なら属性行は表示しない（分類器未ロード/未推論）。
  final Map<String, ({String value, double conf})> attrs;

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: BoxDecoration(
        color: Colors.black54,
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: Colors.white70, width: 1.5),
      ),
      padding: const EdgeInsets.all(6),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Text('#$trackId',
              style: const TextStyle(
                  color: Colors.white,
                  fontSize: 12,
                  fontWeight: FontWeight.bold)),
          const SizedBox(height: 4),
          Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              _tile('cap', cap, capColor),
              const SizedBox(width: 6),
              _tile('label', label, labelColor),
            ],
          ),
          if (attrs.isNotEmpty) ...[
            const SizedBox(height: 4),
            _attrGrid(),
          ],
        ],
      ),
    );
  }

  /// 10属性を 2 列のコンパクト表で出す。信頼度が閾値未満の値は '?' 表示。
  Widget _attrGrid() {
    Widget cell(
        ({
          String key,
          String jp,
          List<String> classes,
          List<String> jpClasses
        }) h) {
      final a = attrs[h.key];
      final bool ok = a != null &&
          a.conf >= AttrAggregate.thresholdFor(h.classes.length);
      final int idx = ok ? h.classes.indexOf(a.value) : -1;
      final String value = idx >= 0 ? h.jpClasses[idx] : '?';
      return SizedBox(
        width: 77,
        child: Text(
          '${h.jp}: $value',
          overflow: TextOverflow.ellipsis,
          style: TextStyle(
            color: ok ? Colors.white : Colors.white38,
            fontSize: 9,
          ),
        ),
      );
    }

    const heads = AttrSchema.heads;
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        for (var i = 0; i < heads.length; i += 2)
          Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              cell(heads[i]),
              if (i + 1 < heads.length) cell(heads[i + 1]),
            ],
          ),
      ],
    );
  }

  Widget _tile(String title, ui.Image? image, Color color) {
    return Container(
      width: 74,
      height: 92,
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(6),
        border: Border.all(color: color, width: 2),
      ),
      padding: const EdgeInsets.all(3),
      child: Column(
        children: [
          Expanded(
            child: image != null
                ? RawImage(image: image, fit: BoxFit.contain)
                : const Center(
                    child: Text('—',
                        style: TextStyle(color: Colors.white38, fontSize: 18)),
                  ),
          ),
          const SizedBox(height: 2),
          Text(title,
              style: const TextStyle(color: Colors.white, fontSize: 10)),
        ],
      ),
    );
  }
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
