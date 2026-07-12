import 'dart:typed_data';
import 'dart:ui' as ui;

import 'attr_classifier.dart';
import 'detector.dart';

/// 追跡中の 1 ボトル。部位画像（capImg/labelImg）はビュー側が所有し、
/// トラック削除時にビュー側で dispose する。
class Track {
  Track(this.id, this.rect);

  final int id;

  /// 最新のボトル位置（見失ったフレームでは前回位置を保持）。
  ui.Rect rect;

  int missing = 0;

  /// 直近フレームでマッチしたボトル検出（ハイライト/ID 表示用。未マッチは null）。
  Detection? lastMatch;

  ui.Image? capImg;
  ui.Image? labelImg;

  /// 10属性のトラック内時間集約（属性分類器が有効な場合のみ更新）。
  final AttrAggregate attrs = AttrAggregate();

  /// 属性推論のレート制限用（トラックごとに間隔を空ける）。
  DateTime lastAttrAt = DateTime.fromMillisecondsSinceEpoch(0);

  void disposeImages() {
    capImg?.dispose();
    capImg = null;
    labelImg?.dispose();
    labelImg = null;
  }
}

/// タップで選択した複数ボトルをフレーム間で追跡するロジック
/// （web / mobile 共用）。座標はすべてモデル入力解像度のピクセル。
class MultiTracker {
  final List<Track> tracks = [];
  int _nextId = 1;

  /// これだけ連続で見失ったらトラック破棄（検出レート依存で数秒相当）。
  static const int _maxMissing = 15;

  /// 引き継ぎに必要な最小 IoU。
  static const double _minIou = 0.1;

  bool get active => tracks.isNotEmpty;

  /// [p] を含む既存トラック（複数なら最小面積のもの）。
  Track? trackAt(ui.Offset p) {
    Track? best;
    for (final t in tracks) {
      if (!t.rect.contains(p)) continue;
      if (best == null ||
          t.rect.width * t.rect.height < best.rect.width * best.rect.height) {
        best = t;
      }
    }
    return best;
  }

  /// [p] を含む最小面積のボトル検出を新規トラックとして追加する。
  /// 該当ボトルがなければ false。
  bool addAt(ui.Offset p, List<Detection> dets) {
    Detection? best;
    for (final d in dets) {
      if (d.cls != 0 || !d.rect.contains(p)) continue;
      if (best == null ||
          d.rect.width * d.rect.height < best.rect.width * best.rect.height) {
        best = d;
      }
    }
    if (best == null) return false;
    final t = Track(_nextId++, best.rect)..lastMatch = best;
    tracks.add(t);
    return true;
  }

  void remove(Track t) => tracks.remove(t);

  /// 全トラックを外して返す（画像の dispose は呼び出し側で行う）。
  List<Track> clear() {
    final removed = List<Track>.of(tracks);
    tracks.clear();
    return removed;
  }

  /// 新しい検出結果で全トラックを更新する。
  /// IoU 降順の貪欲マッチングで 1 検出は 1 トラックにのみ割り当てる。
  /// 戻り値はロストにより破棄されたトラック（画像の dispose は呼び出し側）。
  List<Track> update(List<Detection> dets) {
    for (final t in tracks) {
      t.lastMatch = null;
    }
    final bottles = [for (final d in dets) if (d.cls == 0) d];

    // (iou, track, det) を全組み合わせで並べ、IoU 降順に割り当て
    final pairs = <({double iou, Track track, Detection det})>[];
    for (final t in tracks) {
      for (final d in bottles) {
        final v = iou(t.rect, d.rect);
        if (v >= _minIou) pairs.add((iou: v, track: t, det: d));
      }
    }
    pairs.sort((a, b) => b.iou.compareTo(a.iou));
    final usedTracks = <Track>{};
    final usedDets = Set<Detection>.identity();
    for (final p in pairs) {
      if (usedTracks.contains(p.track) || usedDets.contains(p.det)) continue;
      usedTracks.add(p.track);
      usedDets.add(p.det);
      p.track
        ..rect = p.det.rect
        ..missing = 0
        ..lastMatch = p.det;
    }

    final removed = <Track>[];
    tracks.removeWhere((t) {
      if (t.lastMatch == null && ++t.missing > _maxMissing) {
        removed.add(t);
        return true;
      }
      return false;
    });
    return removed;
  }

  /// クラス [cls] の部位検出を各トラックのボトル枠へ排他的に割り当てる。
  ///
  /// 候補はボトル枠（1 割の余裕つき）に中心がある検出。枠が重なった隣の
  /// ボトルの部位を拾わないよう「部位 bbox の枠内包含率 → スコア」の降順で
  /// 貪欲に対応付け、1 検出は 1 トラックにのみ割り当てる。
  /// 見失い中のトラック（lastMatch == null）は古い位置に他ボトルの部位が
  /// 入り込みうるため対象外。
  static Map<Track, Detection> assignParts(
      List<Track> tracks, List<Detection> dets, int cls) {
    final pairs =
        <({double contain, double score, Track track, Detection det})>[];
    for (final t in tracks) {
      if (t.lastMatch == null) continue;
      final area = t.rect.inflate(t.rect.shortestSide * 0.1);
      for (final d in dets) {
        if (d.cls != cls || !area.contains(d.center)) continue;
        final inter = area.intersect(d.rect);
        final partArea = d.rect.width * d.rect.height;
        final contain =
            partArea <= 0 || inter.width <= 0 || inter.height <= 0
                ? 0.0
                : inter.width * inter.height / partArea;
        pairs.add((contain: contain, score: d.score, track: t, det: d));
      }
    }
    pairs.sort((a, b) {
      final c = b.contain.compareTo(a.contain);
      return c != 0 ? c : b.score.compareTo(a.score);
    });
    final assigned = Map<Track, Detection>.identity();
    final usedDets = Set<Detection>.identity();
    for (final p in pairs) {
      if (assigned.containsKey(p.track) || usedDets.contains(p.det)) continue;
      assigned[p.track] = p.det;
      usedDets.add(p.det);
    }
    return assigned;
  }

  static double iou(ui.Rect a, ui.Rect b) {
    final inter = a.intersect(b);
    if (inter.width <= 0 || inter.height <= 0) return 0;
    final ia = inter.width * inter.height;
    return ia / (a.width * a.height + b.width * b.height - ia);
  }
}

/// [rgba]（s×s の RGBA バッファ）から [rect] を余白付きで切り出す。
/// 小さすぎる場合は null。
({Uint8List rgba, int width, int height})? cropRgba(
    Uint8List rgba, int s, ui.Rect rect) {
  final double pad = rect.shortestSide * 0.15 + 2;
  final int x1 = (rect.left - pad).floor().clamp(0, s - 1);
  final int y1 = (rect.top - pad).floor().clamp(0, s - 1);
  final int x2 = (rect.right + pad).ceil().clamp(x1 + 1, s);
  final int y2 = (rect.bottom + pad).ceil().clamp(y1 + 1, s);
  final int w = x2 - x1;
  final int h = y2 - y1;
  if (w < 4 || h < 4) return null;
  final out = Uint8List(w * h * 4);
  for (int y = 0; y < h; y++) {
    final int src = ((y1 + y) * s + x1) * 4;
    out.setRange(y * w * 4, (y + 1) * w * 4, rgba, src);
  }
  return (rgba: out, width: w, height: h);
}
