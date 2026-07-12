#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""bottle 1クラスのデータに、SAM3 で cap / label を**部位として分離**し、
3クラス(bottle=1, cap=2, label=3) の COCO instance segmentation に拡張する。

各 bottle インスタンスについて、その領域を crop して SAM3 のテキスト検出で cap/label の
top マスクを取り、全体画像座標へ戻して別インスタンス(category 2/3)として追加する。
追加分は parent_bottle_id と seg_source="sam3_part" を持つ。bottle(cat1) は温存。

入力: instances_*_sam3merge.json（bottle）
出力: instances_*_sam3parts.json

  python segment_parts.py --data-root ./bottle --part-min 96       # RunPod
  docker compose --profile tools run --rm seg python segment_parts.py --part-min 160 --limit 12  # local 試し
"""
from __future__ import annotations
import argparse, json, math, os
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm.auto import tqdm

HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = HERE / "datasets" / "bottle"
SRC_ALL = "instances_all_sam3merge.json"
SRC_SPLITS = ["instances_train_sam3merge.json", "instances_val_sam3merge.json", "instances_test_sam3merge.json"]
SUF_FROM, SUF_TO = "_sam3merge", "_sam3parts"

CAP_PROMPTS = ["bottle cap", "lid"]
LABEL_PROMPTS = ["label", "product label"]
CATS = [{"id": 1, "name": "bottle", "supercategory": "bottle"},
        {"id": 2, "name": "cap", "supercategory": "bottle"},
        {"id": 3, "name": "label", "supercategory": "bottle"}]
# 部位ごとの (score閾値, crop面積に対する最小/最大割合)
PART_CFG = {"cap": dict(cat=2, prompts=CAP_PROMPTS, score=0.4, amin=0.003, amax=0.5),
            "label": dict(cat=3, prompts=LABEL_PROMPTS, score=0.4, amin=0.01, amax=0.85)}
POLY_EPS = 0.0015


def hf_login():
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
            from huggingface_hub import login; login(token=token, add_to_git_credential=False)
        except Exception:
            pass


def mask_to_polys(mask):
    import cv2
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = mask.shape[:2]; eps = max(1.0, math.sqrt(h * h + w * w) * POLY_EPS)
    out = []
    for c in cnts:
        if len(c) < 3:
            continue
        ap = cv2.approxPolyDP(c, eps, True).reshape(-1, 2).astype(float)
        if len(ap) >= 3:
            out.append(ap)
    return out


def main():
    import cv2, torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--part-min", type=int, default=96, help="この px 未満の bottle は cap/label を付けない")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--preview", type=int, default=16)
    ap.add_argument("--num-shards", type=int, default=1, help="並列分散の総ワーカー数")
    ap.add_argument("--shard", type=int, default=0, help="このワーカーの番号 (0..num_shards-1)")
    ap.add_argument("--emit-parts", default=None,
                    help="指定時: 新規 part アノテのリストだけを JSON 出力（マージは merge_parts.py）")
    args = ap.parse_args()
    root = args.data_root; ann_dir = root / "annotations"

    coco = json.load(open(ann_dir / SRC_ALL, encoding="utf-8"))
    imgs = {i["id"]: i for i in coco["images"]}
    bottles = [a for a in coco["annotations"]
               if not isinstance(a.get("segmentation"), dict)
               and max(a["bbox"][2], a["bbox"][3]) >= args.part_min]
    if args.limit:
        bottles = bottles[: args.limit]
    if args.num_shards > 1:
        bottles = bottles[args.shard::args.num_shards]
        print(f"[shard] {args.shard}/{args.num_shards}: {len(bottles)} bottles")
    by_img = {}
    for a in bottles:
        by_img.setdefault(a["image_id"], []).append(a)
    print(f"[parts] bottles>= {args.part_min}px: {len(bottles)} in {len(by_img)} images")

    hf_login()
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__() if torch.cuda.is_available() else None
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_sam3_image_model(device=dev)

    new_anns = []
    next_id = max((a["id"] for a in coco["annotations"]), default=0) + 1
    preview = []

    def best_part(proc, pil, cfg, crop_area):
        best = None; best_s = 0.0
        for pr in cfg["prompts"]:
            st = proc.set_image(pil)
            st = proc.set_text_prompt(prompt=pr, state=st)
            m = st.get("masks"); s = st.get("scores")
            if m is None or len(m) == 0:
                continue
            m = (m.float().cpu().numpy()); s = s.float().cpu().numpy().reshape(-1)
            if m.ndim == 4:
                m = m[:, 0]
            for i in range(len(m)):
                mk = m[i] > 0.5; area = int(mk.sum())
                if s[i] < cfg["score"]:
                    continue
                if not (cfg["amin"] * crop_area <= area <= cfg["amax"] * crop_area):
                    continue
                if s[i] > best_s:
                    best_s = float(s[i]); best = mk
        return best, best_s

    for iid, blist in tqdm(by_img.items(), desc="cap/label"):
        path = root / imgs[iid]["file_name"]
        try:
            full = Image.open(path).convert("RGB")
        except Exception:
            continue
        W, H = full.size
        proc = Sam3Processor(model, device=dev, confidence_threshold=0.2)
        for b in blist:
          try:
            x, y, w, h = b["bbox"]; pad = 0.15 * max(w, h)
            x0, y0 = max(0, int(x - pad)), max(0, int(y - pad))
            x1, y1 = min(W, int(x + w + pad)), min(H, int(y + h + pad))
            if x1 - x0 < 5 or y1 - y0 < 5:   # 画像外/退化 bbox はスキップ（crop が例外を投げる）
                continue
            crop = full.crop((x0, y0, x1, y1))
            cw, ch = crop.size
            if cw < 5 or ch < 5:
                continue
            up = 1.0
            pil = crop
            if max(cw, ch) < 320:
                up = 320 / max(cw, ch)
                pil = crop.resize((int(cw * up), int(ch * up)))
            pw, ph = pil.size
            crop_area = pw * ph
            found = {}
            for part, cfg in PART_CFG.items():
                mk, sc = best_part(proc, pil, cfg, crop_area)
                if mk is None:
                    continue
                # pil座標 -> 全体画像座標 でポリゴン化
                segs = []
                for poly in mask_to_polys(mk):
                    poly[:, 0] = poly[:, 0] / up + x0
                    poly[:, 1] = poly[:, 1] / up + y0
                    flat = poly.reshape(-1).tolist()
                    if len(flat) >= 6:
                        segs.append(flat)
                if not segs:
                    continue
                xs = np.concatenate([np.array(s[0::2]) for s in segs])
                ys = np.concatenate([np.array(s[1::2]) for s in segs])
                bx = [float(xs.min()), float(ys.min()), float(xs.max() - xs.min()), float(ys.max() - ys.min())]
                new_anns.append({"id": next_id, "image_id": iid, "category_id": cfg["cat"],
                                 "segmentation": segs, "area": float(int(mk.sum()) / (up * up)),
                                 "bbox": bx, "iscrowd": 0, "seg_source": "sam3_part",
                                 "parent_bottle_id": b["id"], "score": round(sc, 3)})
                found[part] = (next_id, mk, (x0, y0, up))
                next_id += 1
            if found:
                preview.append((iid, b, found))
          except Exception:
            continue

    print(f"[parts] added cap/label instances: {len(new_anns)}")

    # シャードモード: 新規 part アノテのリストだけ出力して終了（id はマージ側で再採番）
    if args.emit_parts:
        json.dump(new_anns, open(args.emit_parts, "w", encoding="utf-8"), ensure_ascii=False)
        print(f"[emit] {args.emit_parts} ({len(new_anns)} parts)")
        return

    # 書き出し（bottle温存 + cap/label追加, categories=3クラス）
    img_split = {}
    for sp in SRC_SPLITS:
        p = ann_dir / sp
        if p.exists():
            for im in json.load(open(p, encoding="utf-8"))["images"]:
                img_split[im["id"]] = sp

    def write(src, add_anns):
        d = json.load(open(ann_dir / src, encoding="utf-8"))
        d["categories"] = CATS
        d["annotations"].extend(add_anns)
        out = ann_dir / src.replace(SUF_FROM, SUF_TO)
        json.dump(d, open(out, "w", encoding="utf-8"), ensure_ascii=False)
        print("[write]", out, "(+%d)" % len(add_anns))

    write(SRC_ALL, new_anns)
    by_split = {}
    for a in new_anns:
        sp = img_split.get(a["image_id"])
        if sp:
            by_split.setdefault(sp, []).append(a)
    for sp in SRC_SPLITS:
        if (ann_dir / sp).exists():
            write(sp, by_split.get(sp, []))

    # QA プレビュー（bottle=青 / cap=赤 / label=緑）
    if args.preview and preview:
        qa = root / "qa_sam3" / "parts"; qa.mkdir(parents=True, exist_ok=True)
        import random
        random.Random(0).shuffle(preview)
        for k, (iid, b, found) in enumerate(preview[: args.preview]):
            arr = np.fromfile(str(root / imgs[iid]["file_name"]), np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            x, y, w, h = [int(v) for v in b["bbox"]]
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 120, 0), 2)
            for part, (aid, mk, (ox, oy, up)) in found.items():
                color = (0, 0, 255) if part == "cap" else (0, 255, 0)
                fmk = np.zeros(img.shape[:2], bool)
                ys, xs = np.where(mk)
                gx = (xs / up + ox).astype(int).clip(0, img.shape[1] - 1)
                gy = (ys / up + oy).astype(int).clip(0, img.shape[0] - 1)
                fmk[gy, gx] = True
                ov = img.copy(); ov[fmk] = color
                img = cv2.addWeighted(ov, 0.5, img, 0.5, 0)
            pad = int(0.3 * max(w, h))
            cx0, cy0 = max(0, x - pad), max(0, y - pad)
            cx1, cy1 = min(img.shape[1], x + w + pad), min(img.shape[0], y + h + pad)
            sub = img[cy0:cy1, cx0:cx1]
            ok, bts = cv2.imencode(".jpg", sub); bts.tofile(str(qa / f"part_{k:03d}.jpg"))
        print("[qa] ->", qa, "(青=bottle 赤=cap 緑=label)")


if __name__ == "__main__":
    main()
