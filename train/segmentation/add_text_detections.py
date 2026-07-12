#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""B案: 既存ラベルは温存し、SAM3 のテキスト検出(grounding)で
『既存 box と重複しない新規ボトル』だけを追加する。

入力 : instances_*_sam3seg.json（box-prompt で精緻化済みの既存データ）
出力 : instances_*_sam3merge.json（既存 + 新規テキスト検出）
新規アノテーションには seg_source="sam3_text", score を付与する。

  python add_text_detections.py --limit 12        # 動作確認（既存アノテ多い画像順）
  python add_text_detections.py                   # 全12,449画像
"""
from __future__ import annotations
import argparse, json, math, random
from pathlib import Path
import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
DATASET_ROOT = HERE / "datasets" / "bottle"
ANN_DIR = DATASET_ROOT / "annotations"

SRC_ALL = "instances_all_sam3seg.json"
SRC_SPLITS = ["instances_train_sam3seg.json",
              "instances_val_sam3seg.json",
              "instances_test_sam3seg.json"]
OUT_SUFFIX_FROM = "_sam3seg"
OUT_SUFFIX_TO = "_sam3merge"

PROMPT = "bottle"
SCORE_THR = 0.4
DEDUP_IOU_EXISTING = 0.5     # 既存 box とこれ以上重なれば「既ラベル」とみなしスキップ
NMS_IOU = 0.7                # 新規検出同士の重複除去
MIN_MASK_AREA = 16
MAX_AREA_RATIO = 0.95
POLY_EPSILON_RATIO = 0.0015
CATEGORY_ID = 1


def imread_u(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def imwrite_u(path, img):
    ok, buf = cv2.imencode(Path(path).suffix, img)
    if ok:
        buf.tofile(str(path))


def hf_login():
    import os
    token = os.environ.get("HF_TOKEN")
    if not token:
        for p in [HERE, *HERE.parents]:
            f = p / ".env"
            if f.exists():
                for line in f.read_text(encoding="utf-8").splitlines():
                    if line.strip().startswith("HF_TOKEN="):
                        token = line.split("=", 1)[1].strip().strip("\"'")
            if token:
                break
    if token:
        os.environ["HF_TOKEN"] = token
        try:
            from huggingface_hub import login
            login(token=token, add_to_git_credential=False)
        except Exception:
            pass


def _np(t):
    if t is None:
        return None
    if hasattr(t, "detach"):
        return t.detach().float().cpu().numpy()
    return np.asarray(t)


def xywh_to_xyxy(b):
    x, y, w, h = b
    return [x, y, x + w, y + h]


def box_iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = (a[2] - a[0]) * (a[3] - a[1])
    ab = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (aa + ab - inter + 1e-9)


def mask_to_bbox(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    return [float(x1), float(y1), float(x2 - x1 + 1), float(y2 - y1 + 1)]


def mask_to_polygons(mask):
    m = mask.astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = mask.shape[:2]
    eps = max(1.0, math.sqrt(h * h + w * w) * POLY_EPSILON_RATIO)
    out = []
    for c in cnts:
        if len(c) < 3:
            continue
        ap = cv2.approxPolyDP(c, eps, True).reshape(-1).astype(float).tolist()
        if len(ap) >= 6:
            out.append(ap)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--score", type=float, default=SCORE_THR)
    ap.add_argument("--no-preview", action="store_true")
    args = ap.parse_args()

    coco = json.load(open(ANN_DIR / SRC_ALL, encoding="utf-8"))
    imgs = {i["id"]: i for i in coco["images"]}
    existing_by_img = {}
    for a in coco["annotations"]:
        existing_by_img.setdefault(a["image_id"], []).append(a)

    # image_id -> split
    img_split = {}
    for sp in SRC_SPLITS:
        d = json.load(open(ANN_DIR / sp, encoding="utf-8"))
        name = sp.replace("instances_", "").replace(OUT_SUFFIX_FROM + ".json", "")
        for im in d["images"]:
            img_split[im["id"]] = name

    order = list(imgs)
    if args.limit is not None:    # 動作確認は既存アノテ多い画像順
        order.sort(key=lambda i: -len(existing_by_img.get(i, [])))
        order = order[: args.limit]

    next_id = max((a["id"] for a in coco["annotations"]), default=0) + 1

    hf_login()
    import torch
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    from PIL import Image
    from tqdm.auto import tqdm

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    model = build_sam3_image_model(device=device)
    processor = Sam3Processor(model, device=device, confidence_threshold=args.score)
    print(f"[sam3] grounding ready on {device}; prompt='{args.prompt}' score>={args.score} "
          f"dedup_iou={DEDUP_IOU_EXISTING}")

    new_anns = []
    qa_pool = []
    n_det = n_added = n_dup = 0

    for iid in tqdm(order, desc="text-detect add"):
        im = imgs[iid]
        path = DATASET_ROOT / im["file_name"]
        try:
            pil = Image.open(path).convert("RGB")
        except Exception:
            continue
        W, H = pil.size
        img_area = W * H
        st = processor.set_image(pil)
        st = processor.set_text_prompt(prompt=args.prompt, state=st)
        masks = _np(st.get("masks"))
        boxes = _np(st.get("boxes"))
        scores = _np(st.get("scores"))
        if masks is None or len(masks) == 0:
            continue
        if masks.ndim == 4:
            masks = masks[:, 0]
        masks = masks > 0.5
        boxes = boxes if boxes is not None else np.zeros((len(masks), 4))
        scores = scores.reshape(-1) if scores is not None else np.ones(len(masks))
        n_det += len(masks)

        # スコア降順
        idx = np.argsort(-scores)
        existing_boxes = [xywh_to_xyxy(a["bbox"]) for a in existing_by_img.get(iid, [])]
        kept_boxes = []   # NMS 用（このフレームで採用した new box）
        added_this = []
        for j in idx:
            m = masks[j]
            area = int(m.sum())
            if area < MIN_MASK_AREA or area / img_area > MAX_AREA_RATIO:
                continue
            mbox = mask_to_bbox(m)
            if mbox is None:
                continue
            xyxy = xywh_to_xyxy(mbox)
            # 既存ラベルと重複 → スキップ
            if existing_boxes and max(box_iou(xyxy, e) for e in existing_boxes) >= DEDUP_IOU_EXISTING:
                n_dup += 1
                continue
            # 新規同士の重複(NMS)
            if kept_boxes and max(box_iou(xyxy, k) for k in kept_boxes) >= NMS_IOU:
                continue
            poly = mask_to_polygons(m)
            if not poly:
                continue
            kept_boxes.append(xyxy)
            ann = {
                "id": next_id,
                "image_id": iid,
                "category_id": CATEGORY_ID,
                "segmentation": poly,
                "area": area,
                "bbox": mbox,
                "iscrowd": 0,
                "seg_source": "sam3_text",
                "score": float(scores[j]),
            }
            new_anns.append(ann)
            added_this.append((m, mbox))
            next_id += 1
            n_added += 1
        if added_this:
            qa_pool.append((iid, added_this))

    print(f"\n[done] detections={n_det} added(new)={n_added} skipped(dup w/ existing)={n_dup}")

    # ---- マージして書き出し ----
    def out_name(src):
        return src.replace(OUT_SUFFIX_FROM, OUT_SUFFIX_TO)

    coco["annotations"].extend(new_anns)
    op = ANN_DIR / out_name(SRC_ALL)
    json.dump(coco, open(op, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"[write] {op}  (+{len(new_anns)} new, total {len(coco['annotations'])})")

    # 分割へ振り分け（新規 ann を、その画像が属する split に追加）
    new_by_split = {}
    for a in new_anns:
        sp = img_split.get(a["image_id"], "train")
        new_by_split.setdefault(sp, []).append(a)
    for sp in SRC_SPLITS:
        d = json.load(open(ANN_DIR / sp, encoding="utf-8"))
        name = sp.replace("instances_", "").replace(OUT_SUFFIX_FROM + ".json", "")
        add = new_by_split.get(name, [])
        d["annotations"].extend(add)
        outp = ANN_DIR / out_name(sp)
        json.dump(d, open(outp, "w", encoding="utf-8"), ensure_ascii=False)
        print(f"[write] {outp}  (+{len(add)} new)")

    # ---- QA: 既存(橙) vs 新規テキスト(緑) を重畳 ----
    if not args.no_preview and qa_pool:
        out_dir = DATASET_ROOT / "qa_sam3" / "merge"
        out_dir.mkdir(parents=True, exist_ok=True)
        qa_pool.sort(key=lambda t: -len(t[1]))
        rng = random.Random(0)
        for k, (iid, added) in enumerate(qa_pool[:24]):
            im = imgs[iid]
            img = imread_u(DATASET_ROOT / im["file_name"])
            if img is None:
                continue
            ov = img.copy()
            # 既存（橙）
            for a in existing_by_img.get(iid, []):
                x, y, w, h = a["bbox"]
                cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (40, 150, 255), 2)
            # 新規（緑）
            for m, mbox in added:
                ov[m] = (0.5 * ov[m] + 0.5 * np.array([60, 220, 60])).astype(np.uint8)
                x, y, w, h = mbox
                cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (60, 220, 60), 2)
            cv2.addWeighted(ov, 0.45, img, 0.55, 0, img)
            t = f"existing(orange)={len(existing_by_img.get(iid, []))} new-text(green)={len(added)}"
            cv2.putText(img, t, (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
            cv2.putText(img, t, (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            imwrite_u(out_dir / f"merge_{k:03d}_{Path(im['file_name']).stem}.jpg", img)
        print(f"[qa] -> {out_dir}  (橙=既存, 緑=新規テキスト検出)")


if __name__ == "__main__":
    main()
