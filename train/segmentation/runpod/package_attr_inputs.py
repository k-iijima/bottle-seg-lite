#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""属性フリートの入力（>=vlm-min の bottle を含む画像 + merge JSON）を tar にまとめる。
ホストで実行:

  python runpod/package_attr_inputs.py [--vlm-min 96]

できた attr_inputs.tar は `_attr_fleet.py seed` で pod 0 (hub役) へ SFTP 転送し、
他 pod は dispatch 時に pod 間 scp で取得する。
"""
import argparse, json, tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent / "datasets" / "pet_bottle"
SRC_ALL = "instances_all_sam3merge.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vlm-min", type=int, default=96)
    ap.add_argument("--out", type=Path, default=HERE / "attr_inputs.tar")
    args = ap.parse_args()

    coco = json.load(open(ROOT / "annotations" / SRC_ALL, encoding="utf-8"))
    imgs = {i["id"]: i for i in coco["images"]}
    need = set()
    for a in coco["annotations"]:
        if isinstance(a.get("segmentation"), dict):
            continue
        if max(a["bbox"][2], a["bbox"][3]) >= args.vlm_min:
            need.add(a["image_id"])
    print(f"[package] images with >= {args.vlm_min}px bottles: {len(need)}/{len(imgs)}")

    with tarfile.open(args.out, "w") as t:      # JPEG は圧縮しない（速度優先）
        t.add(ROOT / "annotations" / SRC_ALL, arcname=f"pet_bottle/annotations/{SRC_ALL}")
        for k, iid in enumerate(sorted(need)):
            fn = imgs[iid]["file_name"]
            t.add(ROOT / fn, arcname=f"pet_bottle/{fn}")
            if (k + 1) % 2000 == 0:
                print(f"  {k+1}/{len(need)}")
    print(f"[package] {args.out} ({args.out.stat().st_size/1e9:.2f} GB)")


if __name__ == "__main__":
    main()
