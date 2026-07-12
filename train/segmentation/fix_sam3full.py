#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""inspect_sam3full.py で検出した機械修正可能な問題を _sam3full に適用する。

処理順（依存関係あり）:
  0. bbox サニタイズ — polygon を画像境界へクランプし、bbox が不正（負幅等）または
     はみ出しの場合は polygon 外接矩形で再計算。潰れたアノテ（TACO の完全画像外等）は削除。
  1. bottle 重複 dedup — 同一画像・IoU>=0.9 の bottle 組（LVIS 由来二重登録）。
     マスク面積の大きい方を残し、消す側の parts は残す側へ再リンク、
     属性は残す側が全 unknown なら消す側から引き継ぐ。iscrowd は対象外。
  2. 浮遊 parts 削除 — どの bottle にも包含 0.5 未満のゴミ検出（親が消えた parts 含む）。
  3. 親リンク再割当 — 記録親との包含 <0.5 かつ 最適 bottle との包含 >=0.5 なら付け替え。
  4. parts NMS — 同一画像・同クラス・IoU>=0.9 は score 高い方を残す。
     さらに同一親・同クラスは IoU>=0.3 で NMS（同じキャップの二重検出。目視確認済み）。
     IoU<0.3 の同一親複数 cap は本物/ゴミ混在のため CVAT 送り（削除しない）。
  5. 属性矛盾の unknown 化 — `cap=uncapped なのに cap_color あり/cap part あり` 等は
     VLM と SAM3 のどちらが正しいか機械判定できない（目視で双方向に誤り確認）ため、
     presence と color の両方を unknown に戻して誤教師信号を除去（マスクは温存）。
  6. 散在マルチポリゴン bottle — マスク面積/bbox面積 <0.08（離れた複数本を1アノテに
     まとめた COCO 由来の群れアノテ）は iscrowd=1 化して学習・評価から除外。

all を修正後、既存 split の画像集合で train/val/test を再生成（ann id は温存）。

  python fix_sam3full.py --dry-run      # 件数確認のみ
  python fix_sam3full.py                # 適用
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA_ROOT = HERE / "datasets" / "pet_bottle"
DUP_IOU = 0.90
SAME_PARENT_NMS_IOU = 0.30  # 同一親・同クラス parts の重複判定
CONTAIN_KEEP = 0.5    # 記録親との包含がこれ以上なら現状維持
CONTAIN_RELINK = 0.5  # 別 bottle との包含がこれ以上なら再リンク（未満なら浮遊=削除）


def bbox_iou(a, b):
    ix = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def contain(part, parent):
    ix = max(0.0, min(part[0] + part[2], parent[0] + parent[2]) - max(part[0], parent[0]))
    iy = max(0.0, min(part[1] + part[3], parent[1] + parent[3]) - max(part[1], parent[1]))
    area = part[2] * part[3]
    return ix * iy / area if area > 0 else 0.0


def fix_all(d, log):
    cats = {c["id"]: c["name"] for c in d["categories"]}
    imgs = {i["id"]: i for i in d["images"]}
    anns = d["annotations"]
    by_img = defaultdict(list)
    for a in anns:
        by_img[a["image_id"]].append(a)

    removed = set()      # ann id -> 除去
    relink = {}          # part ann id -> new parent id

    # --- 0) bbox サニタイズ（不正 bbox / はみ出しのみ対象） ---
    n_clamp = n_degen = 0
    for a in anns:
        if a.get("iscrowd"):
            continue
        im = imgs[a["image_id"]]
        W, H = float(im["width"]), float(im["height"])
        x, y, w, h = a["bbox"]
        if w > 0 and h > 0 and x >= 0 and y >= 0 and x + w <= W and y + h <= H:
            continue
        segs = a.get("segmentation") or []
        segs = [[min(W, max(0.0, v)) if k % 2 == 0 else min(H, max(0.0, v))
                 for k, v in enumerate(poly)] for poly in segs]
        xs = [v for p in segs for v in p[0::2]]
        ys = [v for p in segs for v in p[1::2]]
        if not xs or max(xs) - min(xs) <= 1.0 or max(ys) - min(ys) <= 1.0:
            removed.add(a["id"])   # 画像外で潰れたアノテ
            n_degen += 1
            continue
        a["segmentation"] = segs
        a["bbox"] = [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]
        n_clamp += 1
    log("degenerate_removed", n_degen)
    log("bbox_repaired", n_clamp)

    # --- 1) bottle dedup ---
    n_bottle_dup = 0
    parent_remap = {}
    for image_id, group in by_img.items():
        bots = [a for a in group if cats[a["category_id"]] == "bottle" and not a.get("iscrowd")]
        bots.sort(key=lambda a: -a.get("area", 0))
        for i in range(len(bots)):
            if bots[i]["id"] in removed:
                continue
            for j in range(i + 1, len(bots)):
                loser = bots[j]
                if loser["id"] in removed:
                    continue
                if bbox_iou(bots[i]["bbox"], loser["bbox"]) >= DUP_IOU:
                    keeper = bots[i]
                    removed.add(loser["id"])
                    parent_remap[loser["id"]] = keeper["id"]
                    n_bottle_dup += 1
                    ka, la = keeper.get("attributes"), loser.get("attributes")
                    if ka is not None and la is not None and \
                       all(v == "unknown" for v in ka.values()) and \
                       any(v != "unknown" for v in la.values()):
                        keeper["attributes"] = la
                        if loser.get("attr_conf"):
                            keeper["attr_conf"] = loser["attr_conf"]
    log("bottle_dedup_removed", n_bottle_dup)

    # dedup で消えた bottle を親に持つ parts は残った方へ
    for a in anns:
        pid = a.get("parent_bottle_id")
        if pid in parent_remap:
            relink[a["id"]] = parent_remap[pid]

    # --- 2) 浮遊 parts 削除 / 3) 親リンク再割当 ---
    bottles_by_img = defaultdict(list)
    for a in anns:
        if cats[a["category_id"]] == "bottle" and a["id"] not in removed:
            bottles_by_img[a["image_id"]].append(a)
    bottle_bbox = {a["id"]: a["bbox"] for a in anns if cats[a["category_id"]] == "bottle"}
    n_float = n_relink = 0
    for a in anns:
        if cats[a["category_id"]] == "bottle" or a["id"] in removed:
            continue
        pid = relink.get(a["id"], a["parent_bottle_id"])
        cur = contain(a["bbox"], bottle_bbox[pid]) if pid in bottle_bbox and pid not in removed else -1.0
        if cur >= CONTAIN_KEEP:
            continue
        cands = [(contain(a["bbox"], b["bbox"]), b["id"]) for b in bottles_by_img[a["image_id"]]]
        best, bid = max(cands, default=(0.0, None))
        if best >= CONTAIN_RELINK:
            if bid != a["parent_bottle_id"]:
                relink[a["id"]] = bid
                n_relink += 1
        else:
            removed.add(a["id"])
            n_float += 1
    log("floating_parts_removed", n_float)
    log("parts_relinked", n_relink)

    # --- 4) parts NMS（同クラス IoU>=0.9、score 高い方を残す） ---
    n_nms = 0
    for image_id, group in by_img.items():
        for cname in ("cap", "label"):
            g = [a for a in group
                 if cats[a["category_id"]] == cname and a["id"] not in removed]
            g.sort(key=lambda a: -(a.get("score") or 0.0))
            for i in range(len(g)):
                if g[i]["id"] in removed:
                    continue
                for j in range(i + 1, len(g)):
                    if g[j]["id"] in removed:
                        continue
                    if bbox_iou(g[i]["bbox"], g[j]["bbox"]) >= DUP_IOU:
                        removed.add(g[j]["id"])
                        n_nms += 1
    log("parts_nms_removed", n_nms)

    # --- 4b) 同一親・同クラス parts NMS（IoU>=0.3 = 同じ部位の二重検出） ---
    n_same_parent = 0
    parts_by_parent = defaultdict(list)
    for a in anns:
        if cats[a["category_id"]] != "bottle" and a["id"] not in removed:
            pid = relink.get(a["id"], a["parent_bottle_id"])
            parts_by_parent[(pid, a["category_id"])].append(a)
    for g in parts_by_parent.values():
        g.sort(key=lambda a: -(a.get("score") or 0.0))
        for i in range(len(g)):
            if g[i]["id"] in removed:
                continue
            for j in range(i + 1, len(g)):
                if g[j]["id"] in removed:
                    continue
                if bbox_iou(g[i]["bbox"], g[j]["bbox"]) >= SAME_PARENT_NMS_IOU:
                    removed.add(g[j]["id"])
                    n_same_parent += 1
    log("same_parent_nms_removed", n_same_parent)

    # --- 削除・再リンクを確定 ---
    for a in anns:
        if a["id"] in relink and a["id"] not in removed:
            a["parent_bottle_id"] = relink[a["id"]]
    d["annotations"] = [a for a in anns if a["id"] not in removed]
    anns = d["annotations"]

    # --- 5) 属性矛盾の unknown 化 ---
    final_parts = defaultdict(set)
    for a in anns:
        cname = cats[a["category_id"]]
        if cname != "bottle":
            final_parts[a["parent_bottle_id"]].add(cname)
    n_cap_neutral = n_label_neutral = 0
    for a in anns:
        if cats[a["category_id"]] != "bottle":
            continue
        at = a.get("attributes")
        if not at:
            continue
        if at.get("cap") == "uncapped" and (
                at.get("cap_color") not in ("none", "unknown", None)
                or "cap" in final_parts.get(a["id"], set())):
            at["cap"] = "unknown"
            at["cap_color"] = "unknown"
            n_cap_neutral += 1
        if at.get("label") == "unlabeled" and (
                at.get("label_color") not in ("none", "unknown", None)
                or "label" in final_parts.get(a["id"], set())):
            at["label"] = "unknown"
            at["label_color"] = "unknown"
            n_label_neutral += 1
    log("attr_cap_neutralized", n_cap_neutral)
    log("attr_label_neutralized", n_label_neutral)

    # --- 6) 散在マルチポリゴン bottle の iscrowd 化 ---
    n_crowdified = 0
    for a in anns:
        if cats[a["category_id"]] != "bottle" or a.get("iscrowd"):
            continue
        segs = a.get("segmentation") or []
        area = 0.0
        for poly in segs:
            xs, ys = poly[0::2], poly[1::2]
            s = 0.0
            for i in range(len(xs)):
                j = (i + 1) % len(xs)
                s += xs[i] * ys[j] - xs[j] * ys[i]
            area += abs(s) / 2.0
        w, h = a["bbox"][2], a["bbox"][3]
        if w * h > 0 and area / (w * h) < 0.08:
            a["iscrowd"] = 1
            n_crowdified += 1
    log("scattered_bottles_crowdified", n_crowdified)
    return d, removed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=DATA_ROOT)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    ann_dir = args.data_root / "annotations"

    stats = {}
    log = lambda k, v: (stats.__setitem__(k, v), print(f"  {k}: {v}"))
    d = json.load(open(ann_dir / "instances_all_sam3full.json", encoding="utf-8"))
    n_before = len(d["annotations"])
    d, removed = fix_all(d, log)
    print(f"  annotations: {n_before} -> {len(d['annotations'])}")

    if args.dry_run:
        print("[dry-run] 書き込みなし")
        return

    json.dump(d, open(ann_dir / "instances_all_sam3full.json", "w", encoding="utf-8"),
              ensure_ascii=False)
    print("[write] instances_all_sam3full.json")

    # split 再生成: 既存 split の画像集合を使い、修正済み all から引き直す
    fixed_by_id = {a["id"]: a for a in d["annotations"]}
    anns_by_img = defaultdict(list)
    for a in d["annotations"]:
        anns_by_img[a["image_id"]].append(a)
    imgs_by_id = {i["id"]: i for i in d["images"]}
    for s in ("train", "val", "test"):
        path = ann_dir / f"instances_{s}_sam3full.json"
        old = json.load(open(path, encoding="utf-8"))
        img_ids = [i["id"] for i in old["images"]]
        new = dict(old)
        new["images"] = [imgs_by_id[i] for i in img_ids]
        new["annotations"] = [a for i in img_ids for a in anns_by_img[i]]
        json.dump(new, open(path, "w", encoding="utf-8"), ensure_ascii=False)
        print(f"[write] {path.name}  anns={len(old['annotations'])} -> {len(new['annotations'])}")


if __name__ == "__main__":
    main()
