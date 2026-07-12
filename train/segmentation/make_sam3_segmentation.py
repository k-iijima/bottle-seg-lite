#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
既存の COCO bbox データセット (datasets/bottle) の「低品質な segmentation」だけを、
SAM 3 の box プロンプト (predict_inst) で生成し直して COCO instance segmentation 形式に
し直すスクリプト。ローカル GPU (例: RTX 4060 8GB) 想定。

方針:
  - 既存アノテーションには信頼できる bbox があるので、それを SAM 3 の box プロンプトに渡す
    (テキスト検出ではなく box→mask)。1 物体 = 1 マスクで取りこぼし/重複が起きない。
  - 既存の良質なポリゴンは温存し、「低品質なもの」だけ再生成する:
      * rect 状の簡易マスク (最大リングが <=5 点 = ほぼ bbox の矩形)  ... 既定: 再生成
      * 空の segmentation                                              ... 既定: 再生成
      * RLE (このデータセットでは全て iscrowd=1 の crowd 領域)         ... 既定: スキップ
        └ crowd 領域は「個別インスタンスではない」ため box→mask は無意味で、
          RTMDet-Ins 等の学習では iscrowd=1 は無視される。再生成すると壊れるので既定でスキップ。
  - instances_all.json を一度だけ処理し、annotation id 一致で train/val/test に伝播する
    (3 ファイルの ann id は all の部分集合であることを確認済み)。
  - 元ファイルは上書きせず、`_sam3seg` を付けた新ファイルに書き出す (非破壊)。

使い方:
  python make_sam3_segmentation.py --limit 30   # まず少数で動作確認 (プレビュー生成)
  python make_sam3_segmentation.py              # 全件
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import traceback
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm.auto import tqdm

# ----------------------------------------------------------------------------
# 設定
# ----------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
DATASET_ROOT = HERE / "datasets" / "bottle"
ANN_DIR = DATASET_ROOT / "annotations"
QA_DIR = DATASET_ROOT / "qa_sam3"

SOURCE_ANN = "instances_all.json"                 # 親 (superset)。これを処理して下記へ伝播
SPLIT_ANNS = [                                     # ann id 一致で更新を伝播する分割ファイル
    "instances_train.json",
    "instances_val.json",
    "instances_test.json",
]
OUT_SUFFIX = "_sam3seg"                            # 出力ファイル名のサフィックス (非破壊)

# 再生成の対象判定
REGEN_RECTLIKE = True       # 最大リングが RECTLIKE_MAX_POINTS 点以下のポリゴン
RECTLIKE_MAX_POINTS = 5
REGEN_EMPTY = True          # segmentation が空
REGEN_RLE = False           # RLE (=このデータでは crowd)。既定 False。True にすると crowd も処理
SKIP_ISCROWD = True         # iscrowd=1 は対象外 (crowd 保護)

# 出力 segmentation 形式: "polygon" (既存の良質ポリゴンに合わせる) / "rle"
OUTPUT_SEG = "polygon"
POLY_EPSILON_RATIO = 0.0015  # approxPolyDP の簡略化度合い (画像対角線 * これ)
MIN_POLYGON_POINTS = 3       # 1 リング当たり最低頂点数 (COCO は >=3 点 = 6 座標)

# SAM3 マスク採否
MIN_MASK_AREA = 16           # これ未満のマスクは失敗扱い → 元の segmentation を温存
MIN_BOX_IOU = 0.10           # マスク外接矩形と入力 bbox の IoU がこれ未満なら失敗扱い

DEVICE = "cuda"
BATCH_BOXES = 16             # 1 画像内で predict_inst にまとめて渡す box 数の上限 (VRAM 対策)
PREVIEW_COUNT = 16           # QA プレビュー枚数 (--limit 時は対象全部)


# ----------------------------------------------------------------------------
# 対象判定 / 幾何ユーティリティ
# ----------------------------------------------------------------------------
def is_rectlike(seg) -> bool:
    if not isinstance(seg, list) or len(seg) == 0:
        return False
    return max((len(p) // 2) for p in seg) <= RECTLIKE_MAX_POINTS


def needs_regen(ann) -> bool:
    if SKIP_ISCROWD and ann.get("iscrowd", 0) == 1:
        return False
    seg = ann.get("segmentation")
    if isinstance(seg, dict):                       # RLE
        return REGEN_RLE
    if not seg:                                     # 空
        return REGEN_EMPTY
    if REGEN_RECTLIKE and is_rectlike(seg):         # 矩形状の簡易ポリゴン
        return True
    return False


def xywh_to_xyxy(b):
    x, y, w, h = b
    return [x, y, x + w, y + h]


def box_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / (area_a + area_b - inter + 1e-9)


def mask_to_bbox(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    return [float(x1), float(y1), float(x2 - x1 + 1), float(y2 - y1 + 1)]


def mask_to_polygons(mask):
    m = mask.astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = mask.shape[:2]
    eps = max(1.0, math.sqrt(h * h + w * w) * POLY_EPSILON_RATIO)
    polys = []
    for cnt in contours:
        if len(cnt) < MIN_POLYGON_POINTS:
            continue
        approx = cv2.approxPolyDP(cnt, eps, True)
        if len(approx) < MIN_POLYGON_POINTS:
            continue
        poly = approx.reshape(-1).astype(float).tolist()
        if len(poly) >= 6:
            polys.append(poly)
    return polys


def mask_to_rle(mask):
    from pycocotools import mask as mask_utils
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def to_bool_mask(raw):
    m = np.asarray(raw)
    while m.ndim > 2:
        m = m[0]
    if m.dtype != bool:
        m = m > 0.5
    return m


def hf_login():
    """HF_TOKEN を環境変数 or 親ディレクトリの .env から読み、HuggingFace 認証する。
    SAM 3 のチェックポイントはゲート付きのため未認証だとダウンロードに失敗する。"""
    import os
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        for parent in [HERE, *HERE.parents]:
            env = parent / ".env"
            if env.exists():
                for line in env.read_text(encoding="utf-8").splitlines():
                    s = line.strip()
                    if s.startswith("HF_TOKEN="):
                        token = s.split("=", 1)[1].strip().strip("\"'")
                        break
            if token:
                break
    if not token:
        print("[hf] WARNING: HF_TOKEN が見つかりません。ゲート付き checkpoint の取得に失敗する可能性があります。")
        return
    os.environ["HF_TOKEN"] = token            # huggingface_hub が自動参照
    try:
        from huggingface_hub import login
        login(token=token, add_to_git_credential=False)
        print("[hf] authenticated")
    except Exception as e:  # noqa: BLE001
        print(f"[hf] login warning: {e!r}")


# ----------------------------------------------------------------------------
# メイン
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="処理する対象画像の最大数 (動作確認用)。未指定で全件。")
    ap.add_argument("--no-preview", action="store_true", help="QA プレビューを生成しない")
    ap.add_argument("--seg", choices=["polygon", "rle"], default=OUTPUT_SEG)
    args = ap.parse_args()
    out_seg = args.seg

    src_path = ANN_DIR / SOURCE_ANN
    print(f"[load] {src_path}")
    coco = json.load(open(src_path, encoding="utf-8"))
    images_by_id = {im["id"]: im for im in coco["images"]}

    # 対象アノテーションを画像ごとに収集
    targets_by_img = {}
    for ann in coco["annotations"]:
        if needs_regen(ann):
            targets_by_img.setdefault(ann["image_id"], []).append(ann)

    target_img_ids = sorted(targets_by_img)
    if args.limit is not None:
        target_img_ids = target_img_ids[: args.limit]
    n_target_anns = sum(len(targets_by_img[i]) for i in target_img_ids)
    print(f"[target] images={len(target_img_ids)} annotations={n_target_anns} "
          f"(rectlike={REGEN_RECTLIKE}, empty={REGEN_EMPTY}, rle={REGEN_RLE}, "
          f"skip_iscrowd={SKIP_ISCROWD})")
    if not target_img_ids:
        print("再生成対象がありません。設定を確認してください。")
        return

    # ---- SAM 3 ロード ----
    hf_login()
    print("[sam3] loading model (enable_inst_interactivity=True) ...")
    import torch
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    device = DEVICE if torch.cuda.is_available() else "cpu"
    if device != DEVICE:
        print(f"  WARNING: CUDA 未検出。device={device} で実行します (非常に遅くなります)。")
    model = build_sam3_image_model(device=device, enable_inst_interactivity=True)
    processor = Sam3Processor(model, device=device)
    print(f"[sam3] ready on {device}")

    # ---- 推論ループ ----
    updates = {}        # ann_id -> {segmentation, area, bbox, seg_source}
    n_ok = n_fallback = n_err = 0
    preview_pool = []   # (image_path, [updated_ann_view...])

    for img_id in tqdm(target_img_ids, desc="SAM3 box->mask"):
        im_info = images_by_id[img_id]
        img_path = DATASET_ROOT / im_info["file_name"]
        anns = targets_by_img[img_id]
        try:
            pil = Image.open(img_path).convert("RGB")
            W, H = pil.size
            state = processor.set_image(pil)

            boxes_xyxy = np.array([xywh_to_xyxy(a["bbox"]) for a in anns], dtype=np.float32)
            # クランプ
            boxes_xyxy[:, 0::2] = boxes_xyxy[:, 0::2].clip(0, W - 1)
            boxes_xyxy[:, 1::2] = boxes_xyxy[:, 1::2].clip(0, H - 1)

            masks = []
            for s in range(0, len(anns), BATCH_BOXES):
                chunk = boxes_xyxy[s:s + BATCH_BOXES]
                m, _scores, _ = model.predict_inst(
                    state, point_coords=None, point_labels=None,
                    box=chunk, multimask_output=False,
                )
                m = np.asarray(m)
                if m.ndim == 4:           # [N,1,H,W]
                    for i in range(m.shape[0]):
                        masks.append(to_bool_mask(m[i]))
                elif m.ndim == 3 and len(chunk) == 1:   # [1,H,W]
                    masks.append(to_bool_mask(m))
                else:                     # [N,H,W]
                    for i in range(m.shape[0]):
                        masks.append(to_bool_mask(m[i]))

            preview_anns = []
            for ann, mask in zip(anns, masks):
                area = int(mask.sum())
                mbox = mask_to_bbox(mask)
                ok = (
                    area >= MIN_MASK_AREA
                    and mbox is not None
                    and box_iou(xywh_to_xyxy(mbox), xywh_to_xyxy(ann["bbox"])) >= MIN_BOX_IOU
                )
                if ok and out_seg == "polygon":
                    seg = mask_to_polygons(mask)
                    if not seg:
                        ok = False
                elif ok:  # rle
                    seg = mask_to_rle(mask)

                if not ok:
                    n_fallback += 1
                    continue

                updates[ann["id"]] = {
                    "segmentation": seg,
                    "area": area,
                    "bbox": mbox,
                    "seg_source": "sam3",
                }
                n_ok += 1
                preview_anns.append({"bbox": mbox, "mask": mask, "id": ann["id"]})

            if preview_anns:
                preview_pool.append((img_path, preview_anns))
        except Exception as e:  # noqa: BLE001
            n_err += 1
            print(f"\n[ERROR] img_id={img_id} {img_path}: {e!r}")
            traceback.print_exc()

    print(f"\n[done] regenerated={n_ok} fallback(kept original)={n_fallback} errors={n_err}")

    # ---- 親 (all) に反映して書き出し ----
    def apply_updates(coco_obj):
        cnt = 0
        for ann in coco_obj["annotations"]:
            up = updates.get(ann["id"])
            if up:
                ann["segmentation"] = up["segmentation"]
                ann["area"] = up["area"]
                ann["bbox"] = up["bbox"]
                ann["seg_source"] = up["seg_source"]
                cnt += 1
        return cnt

    def out_name(fname):
        p = Path(fname)
        return p.stem + OUT_SUFFIX + p.suffix

    c = apply_updates(coco)
    out_all = ANN_DIR / out_name(SOURCE_ANN)
    json.dump(coco, open(out_all, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"[write] {out_all}  (updated {c} anns)")

    # ---- train/val/test へ伝播 ----
    for split in SPLIT_ANNS:
        sp = ANN_DIR / split
        if not sp.exists():
            print(f"[skip] {sp} が見つかりません")
            continue
        sc = json.load(open(sp, encoding="utf-8"))
        c = apply_updates(sc)
        op = ANN_DIR / out_name(split)
        json.dump(sc, open(op, "w", encoding="utf-8"), ensure_ascii=False)
        print(f"[write] {op}  (updated {c} anns)")

    # ---- QA プレビュー ----
    if not args.no_preview and preview_pool:
        QA_DIR.mkdir(parents=True, exist_ok=True)
        k = len(preview_pool) if args.limit is not None else PREVIEW_COUNT
        rng = random.Random(42)
        rng.shuffle(preview_pool)
        for idx, (img_path, pviews) in enumerate(preview_pool[:k]):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            ov = img.copy()
            for pa in pviews:
                m = pa["mask"]
                color = np.array([rng.randint(60, 255), rng.randint(60, 255),
                                  rng.randint(60, 255)], dtype=np.uint8)
                ov[m] = (0.5 * ov[m] + 0.5 * color).astype(np.uint8)
                x, y, w, h = map(int, pa["bbox"])
                cv2.rectangle(ov, (x, y), (x + w, y + h), (255, 255, 255), 2)
            out = QA_DIR / f"preview_{idx:03d}_{img_path.stem}.jpg"
            cv2.imwrite(str(out), ov)
        print(f"[qa] previews -> {QA_DIR}")

    print("\n完了。RTMDet-Ins では instances_*"+OUT_SUFFIX+".json を ann_file に指定してください。")


if __name__ == "__main__":
    sys.exit(main())
