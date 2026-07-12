#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""シャード並列(attribute_pipeline.py --emit-attrs)が出した attrs_*.json をまとめて、
属性付き instances_*_sam3attr.json を作る。未処理 ann は全属性 unknown で埋める。

  python merge_attrs.py --data-root datasets/bottle --attrs "runpod/attrs/attrs_*.json"
"""
import argparse, glob, json
from collections import Counter
from pathlib import Path

SRC_ALL = "instances_all_sam3merge.json"
SRC_SPLITS = ["instances_train_sam3merge.json", "instances_val_sam3merge.json", "instances_test_sam3merge.json"]
SUF_FROM, SUF_TO = "_sam3merge", "_sam3attr"
ALL_ATTRS = ["material", "cap", "cap_color", "label", "label_color",
             "fill_level", "crushed", "visibility", "orientation", "depiction"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path("datasets/bottle"))
    ap.add_argument("--attrs", default="runpod/attrs/attrs_*.json")
    args = ap.parse_args()
    ann_dir = args.data_root / "annotations"

    merged = {}
    for f in sorted(glob.glob(args.attrs)):
        d = json.load(open(f, encoding="utf-8"))
        dup = merged.keys() & d.keys()
        if dup:
            print(f"  WARN {f}: {len(dup)} ids already seen (シャード重複?)")
        merged.update(d)
        print(f"  {f}: {len(d)} anns")
    print(f"[merge] attributed anns: {len(merged)}")

    unknown = {a: "unknown" for a in ALL_ATTRS}

    def write(src):
        d = json.load(open(ann_dir / src, encoding="utf-8"))
        hit = 0
        for a in d["annotations"]:
            m = merged.get(str(a["id"]))
            if m:
                a["attributes"] = m["attributes"]
                if m.get("attr_conf"):
                    a["attr_conf"] = m["attr_conf"]
                hit += 1
            else:
                a["attributes"] = dict(unknown)
        out = ann_dir / src.replace(SUF_FROM, SUF_TO)
        json.dump(d, open(out, "w", encoding="utf-8"), ensure_ascii=False)
        print(f"[write] {out} (attributed {hit}/{len(d['annotations'])})")

    write(SRC_ALL)
    for s in SRC_SPLITS:
        if (ann_dir / s).exists():
            write(s)

    for attr in ALL_ATTRS:
        c = Counter(m["attributes"].get(attr, "unknown") for m in merged.values())
        print(f"{attr}:", dict(c.most_common()))


if __name__ == "__main__":
    main()
