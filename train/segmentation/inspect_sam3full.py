#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""_sam3full アノテーションの機械的品質検査。

参照整合・幾何・parts 親子関係・属性の論理矛盾・split リーク・重複を検査し、
サマリを表示、怪しい個体を理由付きで qa_sam3full/report.json に出力する。

  python inspect_sam3full.py                 # 全検査
  python inspect_sam3full.py --skip-files    # 画像ファイル存在チェックを省略
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA_ROOT = HERE / "datasets" / "bottle"

ATTR_SCHEMA = {
    "material": {"unknown", "pet", "glass", "can", "other"},
    "cap": {"unknown", "capped", "uncapped"},
    "cap_color": {"unknown", "none", "white", "black", "red", "blue", "green",
                  "yellow", "orange", "silver", "transparent"},
    "label": {"unknown", "labeled", "unlabeled"},
    "label_color": {"unknown", "none", "white", "black", "red", "blue", "green",
                    "yellow", "orange", "brown", "multicolor"},
    "fill_level": {"unknown", "empty", "low", "half", "high", "full"},
    "crushed": {"unknown", "intact", "crushed"},
    "visibility": {"unknown", "full", "occluded"},
    "orientation": {"unknown", "upright", "lying"},
    "depiction": {"unknown", "real", "depicted"},
}
EPS = 2.0          # bbox/polygon の画像境界はみ出し許容 px
BBOX_SEG_TOL = 8.0  # bbox と polygon 外接矩形のずれ許容 px
PART_CONTAIN_MIN = 0.5   # 部位面積のうち親 bottle bbox 内に入るべき最小比率
MASK_FILL_MIN = 0.08     # mask面積/bbox面積 の下限（これ未満はスリバー疑い）
DUP_IOU = 0.90           # 同クラス bbox IoU がこれ以上なら重複疑い


def poly_area(xs, ys):
    n = len(xs)
    s = 0.0
    for i in range(n):
        j = (i + 1) % n
        s += xs[i] * ys[j] - xs[j] * ys[i]
    return abs(s) / 2.0


def bbox_iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def contain_ratio(part, parent):
    """part bbox 面積のうち parent bbox と重なる比率。"""
    px, py, pw, ph = part
    ix = max(0.0, min(px + pw, parent[0] + parent[2]) - max(px, parent[0]))
    iy = max(0.0, min(py + ph, parent[1] + parent[3]) - max(py, parent[1]))
    area = pw * ph
    return (ix * iy) / area if area > 0 else 0.0


def inspect_all(data_root: Path, skip_files: bool):
    ann_dir = data_root / "annotations"
    d = json.load(open(ann_dir / "instances_all_sam3full.json", encoding="utf-8"))
    cats = {c["id"]: c["name"] for c in d["categories"]}
    imgs = {i["id"]: i for i in d["images"]}
    anns = d["annotations"]
    sus = defaultdict(list)  # reason -> [ann info]
    stats = Counter()

    def flag(reason, a, detail=""):
        sus[reason].append({"ann_id": a["id"], "image_id": a["image_id"],
                            "file": imgs[a["image_id"]]["file_name"] if a["image_id"] in imgs else "?",
                            "cat": cats.get(a["category_id"], "?"), "bbox": a.get("bbox"),
                            "detail": detail})

    # --- 1) ID / 参照整合 ---
    ann_ids = Counter(a["id"] for a in anns)
    for i, n in ann_ids.items():
        if n > 1:
            stats["dup_ann_id"] += 1
    img_ids = Counter(i["id"] for i in d["images"])
    stats["dup_image_id"] = sum(1 for n in img_ids.values() if n > 1)
    for a in anns:
        if a["image_id"] not in imgs:
            flag("bad_image_ref", a)
        if a["category_id"] not in cats:
            flag("bad_category", a)

    # --- 2) 画像ファイル実在 / 未参照ファイル ---
    if not skip_files:
        img_dir = data_root / "images" / "all"
        on_disk = {p.name for p in img_dir.iterdir()}
        referenced = {i["file_name"].rsplit("/", 1)[-1] for i in d["images"]}
        stats["missing_image_file"] = len(referenced - on_disk)
        stats["unreferenced_image_file"] = len(on_disk - referenced)
        for name in sorted(referenced - on_disk)[:20]:
            sus["missing_image_file"].append({"file": name})
        for name in sorted(on_disk - referenced)[:20]:
            sus["unreferenced_image_file"].append({"file": name})

    # --- 3) bbox / segmentation 幾何 ---
    by_img = defaultdict(list)
    for a in anns:
        by_img[a["image_id"]].append(a)
        im = imgs[a["image_id"]]
        W, H = im["width"], im["height"]
        x, y, w, h = a["bbox"]
        if w <= 0 or h <= 0:
            flag("degenerate_bbox", a, f"w={w} h={h}")
            continue
        if x < -EPS or y < -EPS or x + w > W + EPS or y + h > H + EPS:
            flag("bbox_out_of_bounds", a, f"img={W}x{H}")
        seg = a.get("segmentation")
        if a.get("iscrowd"):
            stats["crowd_rle"] += 1
            continue
        if not seg:
            flag("empty_segmentation", a)
            continue
        area_sum, minx, miny, maxx, maxy, bad = 0.0, 1e18, 1e18, -1e18, -1e18, False
        for poly in seg:
            if len(poly) < 6 or len(poly) % 2:
                flag("bad_polygon", a, f"len={len(poly)}")
                bad = True
                break
            xs, ys = poly[0::2], poly[1::2]
            area_sum += poly_area(xs, ys)
            minx, miny = min(minx, min(xs)), min(miny, min(ys))
            maxx, maxy = max(maxx, max(xs)), max(maxy, max(ys))
        if bad:
            continue
        if area_sum <= 1.0:
            flag("tiny_mask_area", a, f"area={area_sum:.2f}")
        fill = area_sum / (w * h) if w * h > 0 else 0
        if fill < MASK_FILL_MIN:
            flag("sliver_mask", a, f"fill={fill:.3f}")
        if fill > 1.10:
            flag("mask_larger_than_bbox", a, f"fill={fill:.3f}")
        dx = max(abs(minx - x), abs(miny - y), abs(maxx - (x + w)), abs(maxy - (y + h)))
        if dx > BBOX_SEG_TOL:
            flag("bbox_seg_mismatch", a, f"max_gap={dx:.1f}px")

    # --- 4) parts 親子関係 ---
    bottles = {a["id"]: a for a in anns if cats[a["category_id"]] == "bottle"}
    caps_of = defaultdict(list)
    for a in anns:
        cname = cats[a["category_id"]]
        if cname == "bottle":
            continue
        pid = a.get("parent_bottle_id")
        parent = bottles.get(pid)
        if parent is None:
            flag("orphan_part", a, f"parent={pid}")
            continue
        if parent["image_id"] != a["image_id"]:
            flag("part_parent_image_mismatch", a)
            continue
        cr = contain_ratio(a["bbox"], parent["bbox"])
        if cr < PART_CONTAIN_MIN:
            flag("part_outside_parent", a, f"contain={cr:.2f}")
        if cname == "cap":
            caps_of[pid].append(a)
        s = a.get("score")
        if s is not None and s < 0.4:
            flag("part_low_score", a, f"score={s:.2f}")
    for pid, cs in caps_of.items():
        if len(cs) > 1:
            flag("multiple_caps_on_bottle", cs[0], f"n={len(cs)} parent={pid}")
            stats["bottles_with_multi_cap"] += 1

    # --- 5) 属性の妥当性・論理矛盾 ---
    long_side = lambda a: max(a["bbox"][2], a["bbox"][3])
    parts_by_parent = defaultdict(set)
    for a in anns:
        cname = cats[a["category_id"]]
        if cname != "bottle" and a.get("parent_bottle_id") in bottles:
            parts_by_parent[a["parent_bottle_id"]].add(cname)
    n_attr = 0
    for b in bottles.values():
        at = b.get("attributes")
        if at is None:
            flag("missing_attributes_dict", b)
            continue
        for k, v in at.items():
            if k not in ATTR_SCHEMA:
                flag("unknown_attr_key", b, k)
            elif v not in ATTR_SCHEMA[k]:
                flag("invalid_attr_value", b, f"{k}={v}")
        known = any(v != "unknown" for v in at.values())
        if known:
            n_attr += 1
        # 論理矛盾
        if at.get("cap") == "uncapped" and at.get("cap_color") not in ("none", "unknown", None):
            flag("uncapped_but_cap_color", b, f"cap_color={at['cap_color']}")
        if at.get("label") == "unlabeled" and at.get("label_color") not in ("none", "unknown", None):
            flag("unlabeled_but_label_color", b, f"label_color={at['label_color']}")
        # 付与規則との整合（>=96px なのに全 unknown / <96px なのに値あり）
        if long_side(b) >= 96 and not known:
            flag("large_bottle_all_unknown", b, f"long={long_side(b):.0f}px")
        if long_side(b) < 96 and known:
            flag("small_bottle_has_attrs", b, f"long={long_side(b):.0f}px")
        # VLM属性 vs SAM3 parts の食い違い（ソフト指標）
        if at.get("cap") == "uncapped" and "cap" in parts_by_parent.get(b["id"], set()):
            flag("attr_uncapped_but_cap_part", b)
        if at.get("label") == "unlabeled" and "label" in parts_by_parent.get(b["id"], set()):
            flag("attr_unlabeled_but_label_part", b)
    stats["bottles_attributed"] = n_attr

    # --- 6) 同クラス高IoU重複 ---
    for image_id, group in by_img.items():
        by_cat = defaultdict(list)
        for a in group:
            by_cat[a["category_id"]].append(a)
        for cid, g in by_cat.items():
            for i in range(len(g)):
                for j in range(i + 1, len(g)):
                    iou = bbox_iou(g[i]["bbox"], g[j]["bbox"])
                    if iou >= DUP_IOU:
                        flag("dup_overlap_same_class", g[i],
                             f"iou={iou:.2f} other_ann={g[j]['id']}")
    stats.update({"images": len(imgs), "annotations": len(anns)})
    for name in ("bottle", "cap", "label"):
        stats[f"n_{name}"] = sum(1 for a in anns if cats[a["category_id"]] == name)
    return sus, stats


def inspect_splits(ann_dir: Path):
    """split リーク・all との整合。"""
    out = {}
    names = {}
    for s in ("train", "val", "test"):
        d = json.load(open(ann_dir / f"instances_{s}_sam3full.json", encoding="utf-8"))
        names[s] = {i["file_name"] for i in d["images"]}
        out[f"{s}_images"] = len(d["images"])
        out[f"{s}_anns"] = len(d["annotations"])
    out["leak_train_val"] = len(names["train"] & names["val"])
    out["leak_train_test"] = len(names["train"] & names["test"])
    out["leak_val_test"] = len(names["val"] & names["test"])
    d = json.load(open(ann_dir / "instances_all_sam3full.json", encoding="utf-8"))
    all_names = {i["file_name"] for i in d["images"]}
    union = names["train"] | names["val"] | names["test"]
    out["all_minus_splits"] = len(all_names - union)
    out["splits_minus_all"] = len(union - all_names)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=DATA_ROOT)
    ap.add_argument("--skip-files", action="store_true")
    ap.add_argument("--out", type=Path, default=None,
                    help="report.json 出力先（既定 <data-root>/qa_sam3full/report.json）")
    args = ap.parse_args()

    sus, stats = inspect_all(args.data_root, args.skip_files)
    split_stats = inspect_splits(args.data_root / "annotations")

    print("=== 基本統計 ===")
    for k in ("images", "annotations", "n_bottle", "n_cap", "n_label",
              "bottles_attributed", "crowd_rle"):
        print(f"  {k}: {stats.get(k, 0)}")
    print("=== split 整合 ===")
    for k, v in split_stats.items():
        mark = "  <-- NG" if ("leak" in k or "minus" in k) and v else ""
        print(f"  {k}: {v}{mark}")
    print("=== 検出された疑義（件数） ===")
    if not sus:
        print("  なし")
    for reason in sorted(sus, key=lambda r: -len(sus[r])):
        print(f"  {reason}: {len(sus[reason])}")

    out = args.out or (args.data_root / "qa_sam3full" / "report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"stats": dict(stats), "split": split_stats,
               "suspicious": {k: v for k, v in sus.items()}},
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[write] {out}")


if __name__ == "__main__":
    main()
