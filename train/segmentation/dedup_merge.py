#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""instances_*_sam3merge.json の重複画像(同一 file_name が別 image_id で複数登録)を統合する。

- file_name でグループ化し、代表 image_id を1つ選ぶ（split 優先度 test>val>train で
  評価セットを保護。同 split 内は最小 id）。
- 非代表 id のアノテーションを代表 id へ付け替え、(category_id, bbox, segmentation) が
  完全一致するアノテは1つに統合。
- 各画像は代表 split のみに所属させ、cross-split リーク(同一画像が train と test 等)を解消。
- all + train/val/test を整合して書き換え。元ファイルは predupe_backup/ に退避。

  python dedup_merge.py --data-root datasets/bottle [--dry-run]
"""
import argparse, json, shutil
from collections import defaultdict
from pathlib import Path

ALL_FILE = "instances_all_sam3merge.json"
SPLIT_FILES = {"train": "instances_train_sam3merge.json",
               "val": "instances_val_sam3merge.json",
               "test": "instances_test_sam3merge.json"}
PRIORITY = {"test": 0, "val": 1, "train": 2}   # 小さいほど優先して保持


def norm(fn):
    return fn.replace("\\", "/").rsplit("/", 1)[-1]


def ann_key(a):
    seg = a.get("segmentation")
    seg_s = json.dumps(seg, sort_keys=True)
    bbox = tuple(round(float(x), 1) for x in a.get("bbox", []))
    return (a.get("category_id"), bbox, seg_s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path("datasets/bottle"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    ann_dir = args.data_root / "annotations"

    all_data = json.load(open(ann_dir / ALL_FILE, encoding="utf-8"))
    imgs = all_data["images"]
    anns = all_data["annotations"]

    # image_id -> split （split ファイルから）
    id_split = {}
    for sp, fn in SPLIT_FILES.items():
        p = ann_dir / fn
        if p.exists():
            for im in json.load(open(p, encoding="utf-8"))["images"]:
                id_split[im["id"]] = sp

    # file_name -> [image records]
    by_fn = defaultdict(list)
    for im in imgs:
        by_fn[norm(im["file_name"])].append(im)

    # 代表 id / split を決定し、old_id -> canon_id, canon_id -> split
    remap = {}            # old image_id -> canonical image_id
    canon_split = {}      # canonical image_id -> split
    canon_imgs = []       # 統合後 images
    dup_groups = 0
    for fn, recs in by_fn.items():
        if len(recs) > 1:
            dup_groups += 1
        # 優先度順: split 優先度 -> id 昇順
        recs_sorted = sorted(recs, key=lambda im: (PRIORITY.get(id_split.get(im["id"], "train"), 2), im["id"]))
        canon = recs_sorted[0]
        cid = canon["id"]
        csp = id_split.get(cid, "train")
        canon_split[cid] = csp
        canon_imgs.append(canon)
        for im in recs:
            remap[im["id"]] = cid

    # アノテを代表 id へ付け替え＆完全一致重複を統合
    seen = defaultdict(set)   # canon_id -> set(ann_key)
    new_anns = []
    dropped = 0
    for a in anns:
        cid = remap.get(a["image_id"], a["image_id"])
        k = ann_key(a)
        if k in seen[cid]:
            dropped += 1
            continue
        seen[cid].add(k)
        a = dict(a)
        a["image_id"] = cid
        new_anns.append(a)

    # 出力組み立て
    def pack(images, annotations):
        out = {k: v for k, v in all_data.items() if k not in ("images", "annotations")}
        out["images"] = images
        out["annotations"] = annotations
        return out

    canon_id_set = {im["id"] for im in canon_imgs}
    all_out = pack(canon_imgs, new_anns)

    split_imgs = {sp: [] for sp in SPLIT_FILES}
    for im in canon_imgs:
        split_imgs[canon_split[im["id"]]].append(im)
    split_ann = {sp: [] for sp in SPLIT_FILES}
    for a in new_anns:
        sp = canon_split.get(a["image_id"])
        if sp:
            split_ann[sp].append(a)

    # レポート
    print(f"[in]  all images={len(imgs)} annotations={len(anns)}")
    print(f"[dup] file_name groups with >1 record: {dup_groups}")
    print(f"[out] all images={len(canon_imgs)} annotations={len(new_anns)} "
          f"(removed {len(imgs)-len(canon_imgs)} dup image records, {dropped} dup annotations)")
    for sp in ("train", "val", "test"):
        print(f"      {sp}: images={len(split_imgs[sp])} annotations={len(split_ann[sp])}")

    # リーク検証
    fn_split = defaultdict(set)
    for sp in SPLIT_FILES:
        for im in split_imgs[sp]:
            fn_split[norm(im["file_name"])].add(sp)
    leak = [fn for fn, ss in fn_split.items() if len(ss) > 1]
    print(f"[check] cross-split leakage after dedup: {len(leak)}")

    if args.dry_run:
        print("[dry-run] no files written.")
        return

    backup = ann_dir / "predupe_backup"
    backup.mkdir(exist_ok=True)
    for fn in [ALL_FILE, *SPLIT_FILES.values()]:
        src = ann_dir / fn
        if src.exists() and not (backup / fn).exists():
            shutil.copy2(src, backup / fn)
    print(f"[backup] originals -> {backup}")

    json.dump(all_out, open(ann_dir / ALL_FILE, "w", encoding="utf-8"))
    for sp, fn in SPLIT_FILES.items():
        json.dump(pack(split_imgs[sp], split_ann[sp]), open(ann_dir / fn, "w", encoding="utf-8"))
    print("[write] dedup 完了")


if __name__ == "__main__":
    main()
