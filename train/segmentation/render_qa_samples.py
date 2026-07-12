#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""inspect_sam3full.py の report.json から怪しい個体をサンプル描画する。

各カテゴリごとに数枚、注釈付き crop PNG を qa_sam3full/samples/ に出力。
凡例: 赤=対象アノテのマスク, 黄=記録された親bottle bbox, 水色=幾何的に最適な bottle bbox,
      マゼンタ=重複相手のマスク。
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
ROOT = HERE / "datasets" / "pet_bottle"
OUT = ROOT / "qa_sam3full" / "samples"
N_PER = 4


def contain(part, parent):
    px, py, pw, ph = part
    ix = max(0.0, min(px + pw, parent[0] + parent[2]) - max(px, parent[0]))
    iy = max(0.0, min(py + ph, parent[1] + parent[3]) - max(py, parent[1]))
    return ix * iy / (pw * ph) if pw * ph > 0 else 0


def draw_poly(dr, seg, color, width=3):
    for poly in seg:
        if len(poly) >= 6:
            dr.line(list(zip(poly[0::2], poly[1::2])) + [(poly[0], poly[1])],
                    fill=color, width=width)


def draw_bbox(dr, b, color, width=3):
    x, y, w, h = b
    dr.rectangle([x, y, x + w, y + h], outline=color, width=width)


def crop_around(img, boxes, margin=60):
    xs = [b[0] for b in boxes] + [b[0] + b[2] for b in boxes]
    ys = [b[1] for b in boxes] + [b[1] + b[3] for b in boxes]
    x0 = max(0, int(min(xs) - margin)); y0 = max(0, int(min(ys) - margin))
    x1 = min(img.width, int(max(xs) + margin)); y1 = min(img.height, int(max(ys) + margin))
    return img.crop((x0, y0, x1, y1)), x0, y0


def spread(items, n):
    if len(items) <= n:
        return items
    step = len(items) // n
    return [items[i * step] for i in range(n)]


def main():
    rep = json.load(open(ROOT / "qa_sam3full" / "report.json", encoding="utf-8"))
    d = json.load(open(ROOT / "annotations" / "instances_all_sam3full.json", encoding="utf-8"))
    cats = {c["id"]: c["name"] for c in d["categories"]}
    anns = {a["id"]: a for a in d["annotations"]}
    imgs = {i["id"]: i for i in d["images"]}
    bot_by_img = defaultdict(list)
    for a in d["annotations"]:
        if cats[a["category_id"]] == "bottle":
            bot_by_img[a["image_id"]].append(a)
    OUT.mkdir(parents=True, exist_ok=True)
    sus = rep["suspicious"]

    def load(a):
        name = imgs[a["image_id"]]["file_name"].rsplit("/", 1)[-1]
        return Image.open(ROOT / "images" / "all" / name).convert("RGB")

    def save(img, tag, i, note):
        p = OUT / f"{tag}_{i}.png"
        img.save(p)
        print(f"[{tag}_{i}] {note}")

    # A) 親リンクずれ: 部位(赤) / 記録親(黄) / 最適bottle(水色)
    po = [x for x in sus.get("part_outside_parent", [])
          if re.search(r"contain=0\.0", x["detail"] or "")]
    for i, x in enumerate(spread(po, N_PER)):
        a = anns[x["ann_id"]]
        parent = anns[a["parent_bottle_id"]]
        best = max(bot_by_img[a["image_id"]], key=lambda b: contain(a["bbox"], b["bbox"]))
        img = load(a); dr = ImageDraw.Draw(img)
        draw_poly(dr, a["segmentation"], (255, 40, 40))
        draw_bbox(dr, parent["bbox"], (255, 220, 0))
        draw_bbox(dr, best["bbox"], (0, 220, 255))
        crop, _, _ = crop_around(img, [a["bbox"], parent["bbox"], best["bbox"]])
        save(crop, "mislink", i,
             f"{x['cat']} ann={a['id']} 記録親={parent['id']} 最適={best['id']} {x['file']}")

    # B) どのbottleにも収まらない部位
    orphanish = []
    for x in sus.get("part_outside_parent", []):
        a = anns[x["ann_id"]]
        if max((contain(a["bbox"], b["bbox"]) for b in bot_by_img[a["image_id"]]), default=0) < 0.5:
            orphanish.append(x)
    for i, x in enumerate(spread(orphanish, N_PER)):
        a = anns[x["ann_id"]]
        parent = anns[a["parent_bottle_id"]]
        img = load(a); dr = ImageDraw.Draw(img)
        draw_poly(dr, a["segmentation"], (255, 40, 40))
        draw_bbox(dr, parent["bbox"], (255, 220, 0))
        crop, _, _ = crop_around(img, [a["bbox"], parent["bbox"]])
        save(crop, "floating_part", i, f"{x['cat']} ann={a['id']} {x['file']}")

    # C) 同クラス重複ペア（赤 vs マゼンタ）
    dp = [x for x in sus.get("dup_overlap_same_class", []) if x["cat"] != "bottle"]
    for i, x in enumerate(spread(dp, N_PER)):
        a = anns[x["ann_id"]]
        b = anns[int(re.search(r"other_ann=(\d+)", x["detail"]).group(1))]
        img = load(a); dr = ImageDraw.Draw(img)
        draw_poly(dr, a["segmentation"], (255, 40, 40))
        draw_poly(dr, b["segmentation"], (255, 0, 255))
        crop, _, _ = crop_around(img, [a["bbox"], b["bbox"]])
        save(crop, "dup_part", i, f"{x['cat']} ann={a['id']}+{b['id']} {x['detail']} {x['file']}")

    # D) bottle同士の高IoU重複
    db = [x for x in sus.get("dup_overlap_same_class", []) if x["cat"] == "bottle"]
    for i, x in enumerate(spread(db, N_PER)):
        a = anns[x["ann_id"]]
        b = anns[int(re.search(r"other_ann=(\d+)", x["detail"]).group(1))]
        img = load(a); dr = ImageDraw.Draw(img)
        draw_poly(dr, a.get("segmentation") or [], (255, 40, 40))
        draw_poly(dr, b.get("segmentation") or [], (255, 0, 255))
        crop, _, _ = crop_around(img, [a["bbox"], b["bbox"]])
        save(crop, "dup_bottle", i, f"ann={a['id']}+{b['id']} {x['file']}")

    # E) 属性矛盾: uncapped なのに cap part あり → bottle crop を出す
    for tag, key in (("attr_capconflict", "attr_uncapped_but_cap_part"),
                     ("attr_labelconflict", "attr_unlabeled_but_label_part")):
        items = sus.get(key, [])
        for i, x in enumerate(spread(items, N_PER)):
            b = anns[x["ann_id"]]
            img = load(b); dr = ImageDraw.Draw(img)
            draw_bbox(dr, b["bbox"], (255, 220, 0))
            for a in d["annotations"]:
                if a.get("parent_bottle_id") == b["id"]:
                    draw_poly(dr, a["segmentation"],
                              (255, 40, 40) if cats[a["category_id"]] == "cap" else (255, 0, 255))
            crop, _, _ = crop_around(img, [b["bbox"]])
            at = b["attributes"]
            save(crop, tag, i,
                 f"ann={b['id']} cap={at['cap']}/{at['cap_color']} label={at['label']}/{at['label_color']} {x['file']}")

    # F) 幾何系: sliver / bbox_seg_mismatch / out_of_bounds
    for tag in ("sliver_mask", "bbox_seg_mismatch", "bbox_out_of_bounds"):
        for i, x in enumerate(spread(sus.get(tag, []), 3)):
            a = anns[x["ann_id"]]
            img = load(a); dr = ImageDraw.Draw(img)
            draw_poly(dr, a.get("segmentation") or [], (255, 40, 40))
            draw_bbox(dr, a["bbox"], (255, 220, 0))
            crop, _, _ = crop_around(img, [a["bbox"]])
            save(crop, tag, i, f"{x['cat']} ann={a['id']} {x['detail']} {x['file']}")

    print(f"\n[done] -> {OUT}")


if __name__ == "__main__":
    main()
