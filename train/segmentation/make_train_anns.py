#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""_sam3full から学習用アノテ instances_*_trainready.json を作る。

- `depiction=depicted`（絵・イラスト・印刷・画面上の描写）の bottle とその cap/label を
  **iscrowd=1（ignore領域）** にする（既定）。実物検出器の学習で偽の正例にしないため。
  --depicted keep で無効化、--depicted drop で完全削除も選べる。
- それ以外は _sam3full そのまま（3クラス、リファイン済みマスク、属性付き）。

  python make_train_anns.py --data-root datasets/pet_bottle
"""
import argparse
import json
from collections import Counter
from pathlib import Path

SPLITS = ["train", "val", "test"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path("datasets/pet_bottle"))
    ap.add_argument("--depicted", choices=["ignore", "drop", "keep"], default="ignore")
    args = ap.parse_args()
    ann_dir = args.data_root / "annotations"

    for s in SPLITS:
        d = json.load(open(ann_dir / f"instances_{s}_sam3full.json", encoding="utf-8"))
        cats = {c["id"]: c["name"] for c in d["categories"]}
        depicted = {a["id"] for a in d["annotations"]
                    if cats[a["category_id"]] == "bottle"
                    and (a.get("attributes") or {}).get("depiction") == "depicted"}
        stats = Counter()
        out_anns = []
        for a in d["annotations"]:
            is_dep = (a["id"] in depicted or a.get("parent_bottle_id") in depicted)
            if is_dep and args.depicted == "drop":
                stats["dropped"] += 1
                continue
            if is_dep and args.depicted == "ignore" and not a.get("iscrowd"):
                a = {**a, "iscrowd": 1}
                stats["ignored"] += 1
            out_anns.append(a)
        d["annotations"] = out_anns
        out = ann_dir / f"instances_{s}_trainready.json"
        json.dump(d, open(out, "w", encoding="utf-8"), ensure_ascii=False)
        n_crowd = sum(1 for a in out_anns if a.get("iscrowd"))
        print(f"[OK][{s}] {out.name}  anns={len(out_anns)} "
              f"depicted_bottles={len(depicted)} {dict(stats)} iscrowd合計={n_crowd}")


if __name__ == "__main__":
    main()
