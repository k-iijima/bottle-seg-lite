#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""refine_masks.py のシャード出力（refined_*.json）を instances_*_sam3full.json に適用する。

- segmentation / bbox / area を差し替え、`seg_refined: true` を付ける（seg_source は温存）
- all に適用後、ann id 一致で train/val/test にも伝播
- rejected（旧マスク温存）と old_iou 分布を集計して品質レポートを出す

  python merge_refined.py --data-root datasets/pet_bottle --refined "runpod/refined/refined_*.json"
  python merge_refined.py --dry-run     # 統計のみ
"""
import argparse
import glob
import json
from collections import Counter
from pathlib import Path

SPLITS = ["all", "train", "val", "test"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path("datasets/pet_bottle"))
    ap.add_argument("--refined", default="runpod/refined/refined_*.json")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    ann_dir = args.data_root / "annotations"

    updates, rejected = {}, {}
    for f in sorted(glob.glob(args.refined)):
        d = json.load(open(f, encoding="utf-8"))
        dup = updates.keys() & d["updates"].keys()
        if dup:
            print(f"  WARN {f}: {len(dup)} ids 重複（シャード重複?）")
        updates.update(d["updates"])
        rejected.update(d.get("rejected", {}))
        print(f"  {f}: updates={len(d['updates'])} rejected={len(d.get('rejected', {}))}")
    print(f"[merge] updates={len(updates)} rejected={len(rejected)}")

    # 品質サマリ
    ious = [u["old_iou"] for u in updates.values()]
    hist = Counter("<0.3" if v < 0.3 else "0.3-0.5" if v < 0.5 else
                   "0.5-0.7" if v < 0.7 else "0.7-0.9" if v < 0.9 else ">=0.9"
                   for v in ious)
    print("[old_iou 分布]", {k: hist[k] for k in ("<0.3", "0.3-0.5", "0.5-0.7", "0.7-0.9", ">=0.9")})
    print("[rejected 理由]", Counter(v.split("=")[0] for v in rejected.values()).most_common())
    if args.dry_run:
        print("[dry-run] 書き込みなし")
        return

    for s in SPLITS:
        path = ann_dir / f"instances_{s}_sam3full.json"
        d = json.load(open(path, encoding="utf-8"))
        hit = 0
        for a in d["annotations"]:
            u = updates.get(str(a["id"]))
            if u:
                a["segmentation"] = u["segmentation"]
                a["bbox"] = u["bbox"]
                a["area"] = u["area"]
                a["seg_refined"] = True
                hit += 1
        json.dump(d, open(path, "w", encoding="utf-8"), ensure_ascii=False)
        print(f"[write] {path.name}  refined {hit}/{len(d['annotations'])}")

    # 低 old_iou（大きく変わった）上位は目視サンプル向けに保存
    worst = sorted(((u["old_iou"], k) for k, u in updates.items()))[:500]
    qa = args.data_root / "qa_sam3full" / "refine_low_iou.json"
    json.dump([{"ann_id": int(k), "old_iou": v} for v, k in worst],
              open(qa, "w", encoding="utf-8"), indent=1)
    print(f"[qa] 変化の大きい上位500件 -> {qa}")


if __name__ == "__main__":
    main()
