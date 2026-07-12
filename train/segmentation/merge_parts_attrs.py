#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""_sam3parts（bottle/cap/label 3クラスマスク）と _sam3attr（bottle 10属性）を統合し、
単一の instances_*_sam3full.json を作る。

両系統の bottle アノテーション ID は _sam3merge 由来で完全一致している前提
（一致しない場合はエラーで停止）。属性は bottle アノテーションのみに持たせる。
cap/label は parent_bottle_id で親を参照できるため属性は複製しない。

  python merge_parts_attrs.py --data-root datasets/bottle
"""
import argparse
import json
from pathlib import Path

SPLITS = ["all", "train", "val", "test"]
SUF_PARTS, SUF_ATTR, SUF_OUT = "_sam3parts", "_sam3attr", "_sam3full"
ALL_ATTRS = ["material", "cap", "cap_color", "label", "label_color",
             "fill_level", "crushed", "visibility", "orientation", "depiction"]


def merge_split(ann_dir: Path, split: str) -> None:
    parts = json.load(open(ann_dir / f"instances_{split}{SUF_PARTS}.json", encoding="utf-8"))
    attr = json.load(open(ann_dir / f"instances_{split}{SUF_ATTR}.json", encoding="utf-8"))

    cats = {c["id"]: c["name"] for c in parts["categories"]}
    bottle_ids = {a["id"] for a in parts["annotations"] if cats[a["category_id"]] == "bottle"}
    attr_by_id = {a["id"]: a for a in attr["annotations"]}
    if bottle_ids != set(attr_by_id):
        only_p = len(bottle_ids - set(attr_by_id))
        only_a = len(set(attr_by_id) - bottle_ids)
        raise SystemExit(f"[{split}] bottle ann ID 不一致: parts側のみ {only_p} / attr側のみ {only_a}")

    unknown = {a: "unknown" for a in ALL_ATTRS}
    n_attr = 0
    for a in parts["annotations"]:
        if cats[a["category_id"]] != "bottle":
            continue
        src = attr_by_id[a["id"]]
        a["attributes"] = src.get("attributes", dict(unknown))
        if src.get("attr_conf"):
            a["attr_conf"] = src["attr_conf"]
        if any(v != "unknown" for v in a["attributes"].values()):
            n_attr += 1

    out = ann_dir / f"instances_{split}{SUF_OUT}.json"
    json.dump(parts, open(out, "w", encoding="utf-8"), ensure_ascii=False)
    n_cls = {name: sum(1 for a in parts["annotations"] if cats[a["category_id"]] == name)
             for name in ("bottle", "cap", "label")}
    print(f"[OK][{split}] {out.name}  images={len(parts['images'])} "
          f"bottle={n_cls['bottle']} (属性あり {n_attr}) cap={n_cls['cap']} label={n_cls['label']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path("datasets/bottle"))
    ap.add_argument("--splits", nargs="+", default=SPLITS)
    args = ap.parse_args()
    for s in args.splits:
        merge_split(args.data_root / "annotations", s)


if __name__ == "__main__":
    main()
