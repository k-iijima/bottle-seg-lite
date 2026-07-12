#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""YouTube ステージング(instances_youtube_sam3merge.json)を本体 sam3merge へ統合する。

- 既存(coco/lvis/taco)の instances_all/train/val/test_sam3merge.json に追記。
- image_id / annotation_id は既存最大値からオフセットして衝突回避。
- split 割当は **動画単位**（youtube_<vid>_<frame>.jpg の <vid> で grouping）し、
  同一動画のフレームが train/test に跨らないようにする（近接フレームのリーク防止）。
  既存比率に合わせ 80/10/10（hash(vid)%10: 0-7=train, 8=val, 9=test）。
- 元ファイルは youtube_premerge_backup/ に退避。

  python merge_youtube.py [--data-root datasets/bottle] [--dry-run]
"""
import argparse, hashlib, json, shutil
from collections import Counter
from pathlib import Path

ALL_FILE = "instances_all_sam3merge.json"
SPLIT_FILES = {"train": "instances_train_sam3merge.json",
               "val": "instances_val_sam3merge.json",
               "test": "instances_test_sam3merge.json"}
YT_FILE = "instances_youtube_sam3merge.json"


def vid_of(file_name):
    b = Path(file_name).name                      # youtube_<vid>_<frame>.jpg
    s = b[len("youtube_"):].rsplit(".", 1)[0]     # <vid>_<frame>
    return s.rsplit("_", 1)[0]                     # <vid>


def split_of(vid):
    h = int(hashlib.md5(vid.encode()).hexdigest(), 16) % 10
    return "train" if h <= 7 else ("val" if h == 8 else "test")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path("datasets/bottle"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    ann_dir = args.data_root / "annotations"

    alld = json.load(open(ann_dir / ALL_FILE, encoding="utf-8"))
    splits = {sp: json.load(open(ann_dir / fn, encoding="utf-8")) for sp, fn in SPLIT_FILES.items()}
    yt = json.load(open(ann_dir / YT_FILE, encoding="utf-8"))

    max_img = max((im["id"] for im in alld["images"]), default=0)
    max_ann = max((a["id"] for a in alld["annotations"]), default=0)

    # YouTube image id -> 新 id / split
    new_img, img_split, id_map = [], {}, {}
    split_count = Counter()
    for i, im in enumerate(yt["images"]):
        nid = max_img + 1 + i
        id_map[im["id"]] = nid
        sp = split_of(vid_of(im["file_name"]))
        rec = dict(im); rec["id"] = nid
        new_img.append((sp, rec)); img_split[nid] = sp; split_count[sp] += 1

    new_ann = []
    for j, a in enumerate(yt["annotations"]):
        rec = dict(a); rec["id"] = max_ann + 1 + j; rec["image_id"] = id_map[a["image_id"]]
        new_ann.append((img_split[rec["image_id"]], rec))

    print(f"[in ] all images={len(alld['images'])} anns={len(alld['annotations'])}")
    print(f"[yt ] images={len(yt['images'])} anns={len(yt['annotations'])} "
          f"videos={len({vid_of(im['file_name']) for im in yt['images']})}")
    print(f"[split] " + " ".join(f"{k}={v}" for k, v in sorted(split_count.items())))

    # all へ追記
    alld["images"] += [r for _, r in new_img]
    alld["annotations"] += [r for _, r in new_ann]
    for sp in SPLIT_FILES:
        splits[sp]["images"] += [r for s, r in new_img if s == sp]
        splits[sp]["annotations"] += [r for s, r in new_ann if s == sp]

    print(f"[out] all images={len(alld['images'])} anns={len(alld['annotations'])}")
    for sp in ("train", "val", "test"):
        print(f"      {sp}: images={len(splits[sp]['images'])} anns={len(splits[sp]['annotations'])}")

    if args.dry_run:
        print("[dry-run] no write."); return

    backup = ann_dir / "youtube_premerge_backup"
    backup.mkdir(exist_ok=True)
    for fn in [ALL_FILE, *SPLIT_FILES.values()]:
        if (ann_dir / fn).exists() and not (backup / fn).exists():
            shutil.copy2(ann_dir / fn, backup / fn)
    print(f"[backup] -> {backup}")

    json.dump(alld, open(ann_dir / ALL_FILE, "w", encoding="utf-8"), ensure_ascii=False)
    for sp, fn in SPLIT_FILES.items():
        json.dump(splits[sp], open(ann_dir / fn, "w", encoding="utf-8"), ensure_ascii=False)
    print("[write] 統合完了")


if __name__ == "__main__":
    main()
