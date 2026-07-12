#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""_sam3full の全マスク（bottle/cap/label）を SAM3 box→mask で再生成する（品質リファイン）。

再検出はしない: 既存アノテの bbox をそのまま box プロンプトに使うため、
インスタンスの同一性（ann id / 属性 / parent_bottle_id / 件数）は完全に保たれ、
segmentation / bbox / area だけが置き換わる。

方式:
  - 長辺 >= CROP_THRESH(128px): 全体画像に set_image し box バッチで predict_inst
  - 長辺 <  CROP_THRESH: bbox 周辺を拡大 crop して単発 predict（小物の輪郭精度対策）
  - 長辺 <  MIN_SIDE(24px): 対象外（SAM3 でも改善しないため旧マスク温存）
  - iscrowd / RLE は対象外
安全ガード（不合格は旧マスク温存で reject 記録）:
  - 新マスク面積 >= 16px
  - 新マスク外接矩形 vs 元 bbox の IoU >= MIN_BOX_IOU
  - 新マスク vs 旧ポリゴンマスクの IoU >= MIN_OLD_IOU（別物にすり替わる事故の防止）

シャード分散（RunPod フリート用）:
  python refine_masks.py --data-root pet_bottle --num-shards 24 --shard 0 --emit refined_0.json
ローカル動作確認:
  docker compose --profile tools run --rm seg python refine_masks.py --limit 3 --qa 12
"""
from __future__ import annotations

import argparse
import json
import math
import time
import traceback
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = HERE / "datasets" / "pet_bottle"

MIN_SIDE = 24          # これ未満の長辺は対象外
CROP_THRESH = 128      # これ未満は拡大 crop ルート
CROP_FACTOR = 3.0      # crop 辺 = 長辺 x これ（最低 CROP_MIN）
CROP_MIN = 320
MIN_MASK_AREA = 16
MIN_BOX_IOU = 0.30     # 新マスク外接矩形 vs 元 bbox
MIN_OLD_IOU = 0.20     # 新マスク vs 旧マスク（すり替わり防止）
BATCH_BOXES = 16


def xywh_to_xyxy(b):
    x, y, w, h = b
    return [x, y, x + w, y + h]


def box_iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ub = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (ua + ub - inter + 1e-9)


def mask_to_bbox(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return [float(xs.min()), float(ys.min()),
            float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1)]


def mask_to_polygons(mask, bbox):
    """輪郭抽出。epsilon は物体サイズ比例（大物 2.5px / 小物 0.5px 上限下限）。"""
    diag = math.hypot(bbox[2], bbox[3])
    eps = min(2.5, max(0.5, 0.01 * diag))
    contours, _ = cv2.findContours(mask.astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for cnt in contours:
        if len(cnt) < 3:
            continue
        approx = cv2.approxPolyDP(cnt, eps, True)
        if len(approx) < 3:
            continue
        poly = approx.reshape(-1).astype(float).tolist()
        if len(poly) >= 6:
            polys.append(poly)
    return polys


def rasterize_old(segs, region_xyxy):
    """旧ポリゴンを region ウィンドウ内に描画した bool マスクを返す。"""
    x1, y1, x2, y2 = [int(v) for v in region_xyxy]
    w, h = max(1, x2 - x1), max(1, y2 - y1)
    m = np.zeros((h, w), np.uint8)
    pts = [np.round(np.array(p, dtype=np.float64).reshape(-1, 2)
                    - [x1, y1]).astype(np.int32) for p in segs if len(p) >= 6]
    if pts:
        cv2.fillPoly(m, pts, 1)
    return m.astype(bool)


def old_new_iou(old_segs, new_mask, old_bbox, new_bbox):
    x1 = min(old_bbox[0], new_bbox[0]); y1 = min(old_bbox[1], new_bbox[1])
    x2 = max(old_bbox[0] + old_bbox[2], new_bbox[0] + new_bbox[2])
    y2 = max(old_bbox[1] + old_bbox[3], new_bbox[1] + new_bbox[3])
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(new_mask.shape[1], int(x2) + 1), min(new_mask.shape[0], int(y2) + 1)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    old = rasterize_old(old_segs, (x1, y1, x2, y2))
    new = new_mask[y1:y2, x1:x2]
    inter = np.logical_and(old, new).sum()
    union = np.logical_or(old, new).sum()
    return float(inter) / float(union) if union else 0.0


def to_bool_mask(raw):
    m = np.asarray(raw)
    while m.ndim > 2:
        m = m[0]
    return m > 0.5 if m.dtype != bool else m


def hf_login():
    import os
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        for parent in [HERE, *HERE.parents]:
            env = parent / ".env"
            if env.exists():
                for line in env.read_text(encoding="utf-8").splitlines():
                    if line.strip().startswith("HF_TOKEN="):
                        token = line.split("=", 1)[1].strip().strip("\"'")
                        break
            if token:
                break
    if token:
        os.environ["HF_TOKEN"] = token
        try:
            from huggingface_hub import login
            login(token=token, add_to_git_credential=False)
        except Exception as e:  # noqa: BLE001
            print(f"[hf] login warning: {e!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--ann", default="annotations/instances_all_sam3full.json")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--emit", default=None,
                    help="出力 JSON（既定 refined_<shard>.json）")
    ap.add_argument("--limit", type=int, default=None, help="処理画像数の上限（動作確認）")
    ap.add_argument("--qa", type=int, default=0, help="QA プレビュー枚数")
    args = ap.parse_args()
    emit = Path(args.emit or f"refined_{args.shard}.json")

    coco = json.load(open(args.data_root / args.ann, encoding="utf-8"))
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    imgs = {i["id"]: i for i in coco["images"]}

    targets = defaultdict(list)     # image_id -> [ann]
    n_skip_small = n_skip_crowd = 0
    for a in coco["annotations"]:
        if a.get("iscrowd") or isinstance(a.get("segmentation"), dict):
            n_skip_crowd += 1
            continue
        if max(a["bbox"][2], a["bbox"][3]) < MIN_SIDE:
            n_skip_small += 1
            continue
        targets[a["image_id"]].append(a)

    img_ids = sorted(targets)
    img_ids = [i for k, i in enumerate(img_ids) if k % args.num_shards == args.shard]
    if args.limit:
        img_ids = img_ids[:args.limit]
    n_anns = sum(len(targets[i]) for i in img_ids)
    print(f"[target] shard {args.shard}/{args.num_shards}: images={len(img_ids)} anns={n_anns} "
          f"(skip: small={n_skip_small} crowd/rle={n_skip_crowd})")

    hf_login()
    import torch
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_sam3_image_model(device=device, enable_inst_interactivity=True)
    processor = Sam3Processor(model, device=device)
    autocast = torch.autocast(device, dtype=torch.bfloat16) if device == "cuda" else None
    print(f"[sam3] ready on {device}")

    def predict_boxes(state, boxes_xyxy):
        out = []
        for s in range(0, len(boxes_xyxy), BATCH_BOXES):
            chunk = np.asarray(boxes_xyxy[s:s + BATCH_BOXES], dtype=np.float32)
            if autocast:
                with autocast:
                    m, _sc, _ = model.predict_inst(state, point_coords=None,
                                                   point_labels=None, box=chunk,
                                                   multimask_output=False)
            else:
                m, _sc, _ = model.predict_inst(state, point_coords=None,
                                               point_labels=None, box=chunk,
                                               multimask_output=False)
            m = np.asarray(m)
            if m.ndim == 3 and len(chunk) == 1:
                out.append(to_bool_mask(m))
            else:
                for i in range(m.shape[0]):
                    out.append(to_bool_mask(m[i]))
        return out

    updates, rejected = {}, {}
    n_ok = n_rej = n_err = 0
    qa_pool = []
    t0 = time.time()

    for done, img_id in enumerate(img_ids):
        im = imgs[img_id]
        path = args.data_root / im["file_name"]
        anns = targets[img_id]
        try:
            pil = Image.open(path).convert("RGB")
            W, H = pil.size
            big = [a for a in anns if max(a["bbox"][2], a["bbox"][3]) >= CROP_THRESH]
            small = [a for a in anns if max(a["bbox"][2], a["bbox"][3]) < CROP_THRESH]
            results = []   # (ann, full_mask)

            if big:
                state = processor.set_image(pil)
                boxes = []
                for a in big:
                    b = xywh_to_xyxy(a["bbox"])
                    boxes.append([max(0, b[0]), max(0, b[1]),
                                  min(W - 1, b[2]), min(H - 1, b[3])])
                for a, m in zip(big, predict_boxes(state, boxes)):
                    results.append((a, m))

            for a in small:
                x, y, w, h = a["bbox"]
                side = max(CROP_MIN, CROP_FACTOR * max(w, h))
                cx, cy = x + w / 2, y + h / 2
                x1 = int(max(0, cx - side / 2)); y1 = int(max(0, cy - side / 2))
                x2 = int(min(W, cx + side / 2)); y2 = int(min(H, cy + side / 2))
                crop = pil.crop((x1, y1, x2, y2))
                state = processor.set_image(crop)
                b = [max(0, x - x1), max(0, y - y1),
                     min(x2 - x1 - 1, x + w - x1), min(y2 - y1 - 1, y + h - y1)]
                cm = predict_boxes(state, [b])[0]
                full = np.zeros((H, W), bool)
                full[y1:y1 + cm.shape[0], x1:x1 + cm.shape[1]] = cm
                results.append((a, full))

            for a, mask in results:
                area = int(mask.sum())
                mbox = mask_to_bbox(mask)
                reason = None
                if area < MIN_MASK_AREA or mbox is None:
                    reason = "empty"
                elif box_iou(xywh_to_xyxy(mbox), xywh_to_xyxy(a["bbox"])) < MIN_BOX_IOU:
                    reason = "box_mismatch"
                else:
                    oiou = old_new_iou(a["segmentation"], mask, a["bbox"], mbox)
                    if oiou < MIN_OLD_IOU:
                        reason = f"old_iou={oiou:.2f}"
                if reason is None:
                    segs = mask_to_polygons(mask, mbox)
                    if not segs:
                        reason = "no_polygon"
                if reason:
                    rejected[str(a["id"])] = reason
                    n_rej += 1
                    continue
                updates[str(a["id"])] = {"segmentation": segs, "bbox": mbox,
                                         "area": area, "old_iou": round(oiou, 3)}
                n_ok += 1
                if args.qa and len(qa_pool) < args.qa:
                    qa_pool.append((path, a, mask, mbox))
        except Exception as e:  # noqa: BLE001
            n_err += 1
            print(f"[ERROR] img={img_id} {path}: {e!r}")
            traceback.print_exc()
        if (done + 1) % 50 == 0:
            el = time.time() - t0
            print(f"  {done+1}/{len(img_ids)} imgs  ok={n_ok} rej={n_rej} "
                  f"{(done+1)/el:.2f} img/s", flush=True)

    json.dump({"updates": updates, "rejected": rejected},
              open(emit, "w", encoding="utf-8"))
    print(f"[done] ok={n_ok} rejected={n_rej} errors={n_err} "
          f"({time.time()-t0:.0f}s) -> {emit}")

    if qa_pool:
        qa_dir = args.data_root / "qa_sam3full" / "refine_preview"
        qa_dir.mkdir(parents=True, exist_ok=True)
        for i, (path, a, mask, mbox) in enumerate(qa_pool):
            img = cv2.imread(str(path))
            if img is None:
                continue
            ov = img.copy()
            ov[mask] = (0.5 * ov[mask] + 0.5 * np.array([60, 60, 255])).astype(np.uint8)
            old = rasterize_old(a["segmentation"], (0, 0, img.shape[1], img.shape[0]))
            cv2.drawContours(ov, cv2.findContours(old.astype(np.uint8),
                             cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0],
                             -1, (0, 255, 0), 2)
            x, y, w, h = [int(v) for v in a["bbox"]]
            m = 40
            crop = ov[max(0, y - m):y + h + m, max(0, x - m):x + w + m]
            cv2.imwrite(str(qa_dir / f"refine_{i:02d}_ann{a['id']}.jpg"), crop)
        print(f"[qa] previews -> {qa_dir}  (赤=新マスク, 緑=旧輪郭)")


if __name__ == "__main__":
    main()
