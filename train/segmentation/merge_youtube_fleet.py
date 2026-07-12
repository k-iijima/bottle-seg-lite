#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""yt_fleet が回収した youtube_out_<shard>.tar.gz をまとめ、pod 間グローバル dedup
（CLIP 埋め込みのコサイン類似）を掛けて pet_bottle データセットへ取り込む。

各 pod は frames を別動画から作るので名前衝突はないが、似た構図は pod を跨いで重複し得る。
そこで埋め込みで全体 greedy dedup してから images/all へ配置する。

出力（ステージング・非破壊）:
  datasets/pet_bottle/images/all/youtube_*.jpg
  datasets/pet_bottle/annotations/instances_youtube_sam3merge.json
  datasets/pet_bottle/metadata/manifest_youtube.csv

  python merge_youtube_fleet.py [--fleet-dir runpod/youtube_fleet] [--dedup-thresh 0.9]
本体(instances_*_sam3merge.json)への統合は merge_youtube.py で別途。
"""
import argparse, csv, json, shutil, tarfile, tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATASET_ROOT = HERE / "datasets" / "pet_bottle"
IMG_DIR = DATASET_ROOT / "images" / "all"
ANN_DIR = DATASET_ROOT / "annotations"
META_DIR = DATASET_ROOT / "metadata"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fleet-dir", default=str(HERE / "runpod" / "youtube_fleet"))
    ap.add_argument("--dedup-thresh", type=float, default=0.90,
                    help="pod 間 dedup の CLIP コサイン類似閾値（local より少し緩め）")
    args = ap.parse_args()
    fleet_dir = Path(args.fleet_dir)
    tars = sorted(fleet_dir.glob("youtube_out_*.tar.gz"))
    if not tars:
        print(f"[merge] {fleet_dir} に tarball がありません"); return
    print(f"[merge] {len(tars)} tarballs")

    tmp = Path(tempfile.mkdtemp(prefix="ytmerge_"))
    records = []          # (record_dict, frame_path)
    for t in tars:
        d = tmp / t.stem.replace(".tar", "")
        d.mkdir(parents=True, exist_ok=True)
        with tarfile.open(t) as tf:
            tf.extractall(d)
        pj = next(d.glob("parts_*.json"), None)
        if not pj:
            print("  [warn] parts json なし:", t.name); continue
        recs = json.load(open(pj, encoding="utf-8"))["records"]
        for r in recs:
            fp = d / "frames" / r["file_name"]
            if fp.exists():
                records.append((r, fp))
        print(f"  {t.name}: {len(recs)} records")

    print(f"[merge] 合計 {len(records)} records → グローバル dedup (>= {args.dedup_thresh})")

    # greedy global dedup
    kept = []
    kept_emb = np.zeros((0, 512), dtype=np.float32)
    n_dup = 0
    for r, fp in records:
        e = np.asarray(r["embedding"], dtype=np.float32)
        if kept_emb.shape[0] and float((kept_emb @ e).max()) >= args.dedup_thresh:
            n_dup += 1; continue
        kept.append((r, fp))
        kept_emb = np.vstack([kept_emb, e[None, :]])
    print(f"[merge] dedup 除去 {n_dup} / 採用 {len(kept)}")

    IMG_DIR.mkdir(parents=True, exist_ok=True)
    ANN_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)

    images, annots, manifest_rows = [], [], []
    img_id = ann_id = 0
    for r, fp in kept:
        rel = f"images/all/{r['file_name']}"
        shutil.copy2(fp, DATASET_ROOT / rel)
        images.append({"id": img_id, "file_name": rel,
                       "width": r["width"], "height": r["height"]})
        for a in r["annotations"]:
            a = dict(a); a["id"] = ann_id; a["image_id"] = img_id
            annots.append(a); ann_id += 1
        m = dict(r["manifest"]); m["file_name"] = rel
        manifest_rows.append(m)
        img_id += 1

    coco = {"info": {"description": "youtube CC pet-bottle frames (fleet)"},
            "licenses": [], "images": images, "annotations": annots,
            "categories": [{"id": 1, "name": "bottle", "supercategory": "bottle"}]}
    out_ann = ANN_DIR / "instances_youtube_sam3merge.json"
    json.dump(coco, open(out_ann, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"[write] {out_ann}  images={len(images)} annotations={len(annots)}")

    if manifest_rows:
        mpath = META_DIR / "manifest_youtube.csv"
        with open(mpath, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
            w.writeheader(); w.writerows(manifest_rows)
        print(f"[write] {mpath}")

    shutil.rmtree(tmp, ignore_errors=True)
    print("[merge] 完了。統合は merge_youtube.py で instances_*_sam3merge.json へ。")


if __name__ == "__main__":
    main()
