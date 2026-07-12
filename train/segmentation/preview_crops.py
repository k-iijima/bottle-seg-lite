#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SAM3 で再生成したマスクを目視確認するための「ズーム済みクロップ」を作る。
GPU 不要（polygon を cv2 で描画するだけ）。ローカルでもコンテナでも動く。

  python preview_crops.py                       # 既定: instances_all_sam3seg.json の sam3 再生成分
  python preview_crops.py --ann instances_train_sam3seg.json --n 40
  python preview_crops.py --all                 # seg_source 問わず全アノテーション対象
"""
from __future__ import annotations
import argparse, json, math, random
from pathlib import Path
import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
DATASET_ROOT = HERE / "datasets" / "pet_bottle"
ANN_DIR = DATASET_ROOT / "annotations"


def imread_u(path):
    """Unicode パス対応の読み込み（Windows の非ASCIIパス対策）。"""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def imwrite_u(path, img):
    ok, buf = cv2.imencode(Path(path).suffix, img)
    if ok:
        buf.tofile(str(path))
    return ok


def render_seg(canvas, seg, origin, color):
    """polygon (COCO list 形式) を origin 原点のクロップ座標で描画。"""
    ox, oy = origin
    if isinstance(seg, dict):       # RLE は対象外（このデータでは crowd）。スキップ
        return
    overlay = canvas.copy()
    for ring in seg:
        pts = np.array(ring, dtype=np.float32).reshape(-1, 2)
        pts[:, 0] -= ox
        pts[:, 1] -= oy
        pts = pts.round().astype(np.int32)
        if len(pts) >= 3:
            cv2.fillPoly(overlay, [pts], color.tolist())
            cv2.polylines(canvas, [pts], True, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.45, canvas, 0.55, 0, canvas)


def render_whole(args):
    """SAM3再生成を含む画像について、その画像の全アノテーションを重畳する。
    緑=SAM3再生成 / 橙=元から温存。複数ボトルが全て segmentation 済みであることを確認する用。"""
    coco = json.load(open(ANN_DIR / args.ann, encoding="utf-8"))
    imgs = {i["id"]: i for i in coco["images"]}
    by_img = {}
    for a in coco["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a)
    sam_imgs = [i for i in by_img if any(a.get("seg_source") == "sam3" for a in by_img[i])]
    # ボトル数が多い画像を優先
    sam_imgs.sort(key=lambda i: -len(by_img[i]))
    sam_imgs = sam_imgs[: args.n]
    out_dir = DATASET_ROOT / "qa_sam3" / "whole"
    out_dir.mkdir(parents=True, exist_ok=True)
    GREEN = np.array([60, 220, 60], np.uint8)
    ORANGE = np.array([40, 150, 255], np.uint8)
    made = 0
    for k, iid in enumerate(sam_imgs):
        im = imgs[iid]
        img = imread_u(DATASET_ROOT / im["file_name"])
        if img is None:
            continue
        n_sam = 0
        for a in by_img[iid]:
            is_sam = a.get("seg_source") == "sam3"
            n_sam += is_sam
            render_seg(img, a.get("segmentation"), (0, 0), GREEN if is_sam else ORANGE)
            x, y, w, h = a["bbox"]
            col = (60, 220, 60) if is_sam else (40, 150, 255)
            cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), col, 2)
        cv2.putText(img, f"anns={len(by_img[iid])} sam3={n_sam}", (6, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(img, f"anns={len(by_img[iid])} sam3={n_sam}", (6, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        imwrite_u(out_dir / f"whole_{k:03d}_{Path(im['file_name']).stem}.jpg", img)
        made += 1
    print(f"wrote {made} whole-image overlays -> {out_dir}  (緑=SAM3再生成, 橙=元から温存)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", default="instances_all_sam3seg.json")
    ap.add_argument("--n", type=int, default=40, help="出力クロップ数")
    ap.add_argument("--pad", type=float, default=0.6, help="bbox に対する余白割合")
    ap.add_argument("--size", type=int, default=320, help="クロップ拡大後の一辺(px)")
    ap.add_argument("--all", action="store_true", help="seg_source を問わず対象にする")
    ap.add_argument("--out", default="qa_sam3/crops")
    ap.add_argument("--whole", action="store_true",
                    help="SAM3再生成を含む画像の『全アノテーション』を色分け表示（緑=SAM3, 橙=元）")
    args = ap.parse_args()

    if args.whole:
        return render_whole(args)

    coco = json.load(open(ANN_DIR / args.ann, encoding="utf-8"))
    imgs = {i["id"]: i for i in coco["images"]}
    anns = [a for a in coco["annotations"]
            if (args.all or a.get("seg_source") == "sam3")
            and not isinstance(a.get("segmentation"), dict) and a.get("segmentation")]
    print(f"target anns: {len(anns)} (ann={args.ann}, all={args.all})")
    if not anns:
        print("対象なし"); return

    random.Random(42).shuffle(anns)
    anns = anns[: args.n]
    out_dir = DATASET_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    made = 0
    for k, a in enumerate(anns):
        im = imgs[a["image_id"]]
        img = imread_u(DATASET_ROOT / im["file_name"])
        if img is None:
            continue
        H, W = img.shape[:2]
        x, y, w, h = a["bbox"]
        pad = args.pad * max(w, h)
        x0, y0 = int(max(0, x - pad)), int(max(0, y - pad))
        x1, y1 = int(min(W, x + w + pad)), int(min(H, y + h + pad))
        crop = img[y0:y1, x0:x1].copy()
        if crop.size == 0:
            continue
        color = np.array([60, 220, 60], dtype=np.uint8)  # 緑系
        render_seg(crop, a["segmentation"], (x0, y0), color)
        # bbox も描く
        cv2.rectangle(crop, (int(x - x0), int(y - y0)),
                      (int(x + w - x0), int(y + h - y0)), (0, 165, 255), 1)
        # 拡大
        scale = args.size / max(crop.shape[:2])
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        label = f"id{a['id']} {int(w)}x{int(h)}px"
        cv2.putText(crop, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(crop, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        out = out_dir / f"crop_{k:03d}_{Path(im['file_name']).stem}_id{a['id']}.jpg"
        imwrite_u(out, crop)
        made += 1

    # 一覧モンタージュも作る
    files = sorted(out_dir.glob("crop_*.jpg"))[: min(25, made)]
    if files:
        cols = 5
        tiles = [cv2.resize(imread_u(f), (args.size, args.size)) for f in files]
        while len(tiles) % cols:
            tiles.append(np.zeros((args.size, args.size, 3), np.uint8))
        rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
        imwrite_u(out_dir / "_montage.jpg", np.vstack(rows))
    print(f"wrote {made} crops -> {out_dir}  (montage: _montage.jpg)")


if __name__ == "__main__":
    main()
