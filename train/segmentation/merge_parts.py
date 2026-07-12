#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""シャード並列(segment_parts.py --emit-parts)が出した parts_*.json をまとめて、
3クラス(bottle/cap/label)の instances_*_sam3parts.json を作る。id を再採番して衝突回避。

  python merge_parts.py --data-root ./pet_bottle --parts "parts_*.json"
"""
import argparse, glob, json
from pathlib import Path

CATS = [{"id": 1, "name": "bottle", "supercategory": "bottle"},
        {"id": 2, "name": "cap", "supercategory": "bottle"},
        {"id": 3, "name": "label", "supercategory": "bottle"}]
SRC_ALL = "instances_all_sam3merge.json"
SRC_SPLITS = ["instances_train_sam3merge.json", "instances_val_sam3merge.json", "instances_test_sam3merge.json"]
SUF_FROM, SUF_TO = "_sam3merge", "_sam3parts"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path("pet_bottle"))
    ap.add_argument("--parts", default="parts_*.json")
    args = ap.parse_args()
    ann_dir = args.data_root / "annotations"

    base = json.load(open(ann_dir / SRC_ALL, encoding="utf-8"))
    next_id = max((a["id"] for a in base["annotations"]), default=0) + 1

    parts = []
    for f in sorted(glob.glob(args.parts)):
        lst = json.load(open(f, encoding="utf-8"))
        parts.extend(lst)
        print(f"  {f}: {len(lst)} parts")
    for a in parts:           # id 再採番（シャード間衝突を解消）
        a["id"] = next_id; next_id += 1
    print(f"[merge] total parts: {len(parts)}")

    img_split = {}
    for sp in SRC_SPLITS:
        p = ann_dir / sp
        if p.exists():
            for im in json.load(open(p, encoding="utf-8"))["images"]:
                img_split[im["id"]] = sp

    def write(src, add):
        d = json.load(open(ann_dir / src, encoding="utf-8"))
        d["categories"] = CATS
        d["annotations"].extend(add)
        out = ann_dir / src.replace(SUF_FROM, SUF_TO)
        json.dump(d, open(out, "w", encoding="utf-8"), ensure_ascii=False)
        print("[write]", out, "(+%d)" % len(add))

    write(SRC_ALL, parts)
    by_split = {}
    for a in parts:
        sp = img_split.get(a["image_id"])
        if sp:
            by_split.setdefault(sp, []).append(a)
    for sp in SRC_SPLITS:
        if (ann_dir / sp).exists():
            write(sp, by_split.get(sp, []))


if __name__ == "__main__":
    main()
