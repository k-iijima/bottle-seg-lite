#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""YouTube の Creative Commons 動画から「ペットボトルが映るフレーム」を収集する。

日本のペットボトル関連シーン中心。CC ライセンスのみ DL → 低fpsでフレーム抽出 →
SAM3 text "bottle" でボトル有無を判定（同時に初期 bottle アノテ取得）→
CLIP 埋め込みのコサイン類似で近接フレームを除去（多様性確保）。

出力（非破壊・ステージング）:
  datasets/bottle/images/all/youtube_<vid>_<frame>.jpg
  datasets/bottle/annotations/instances_youtube_sam3merge.json   (COCO: bottle のみ)
  datasets/bottle/metadata/manifest_youtube.csv
  datasets/bottle/qa_youtube/preview_*.jpg
本体(instances_*_sam3merge.json)への統合は merge_youtube.py で別途行う（レビュー後）。

  # 1) DL のみ
  python collect_youtube.py download --max-videos 30
  # 2) 抽出+判定+dedup（GPU）
  python collect_youtube.py process --max-frames 500
  # まとめて:
  python collect_youtube.py all --max-videos 30 --max-frames 500
"""
from __future__ import annotations
import argparse, csv, json, math, os, subprocess, sys, urllib.parse
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
DATASET_ROOT = HERE / "datasets" / "bottle"
IMG_DIR = DATASET_ROOT / "images" / "all"
ANN_DIR = DATASET_ROOT / "annotations"
META_DIR = DATASET_ROOT / "metadata"
QA_DIR = DATASET_ROOT / "qa_youtube"
WORK = HERE / "runpod" / "youtube"        # 動画と中間ファイル（datasets 外）
VID_DIR = WORK / "videos"

# 日本のペットボトル関連シーン中心の検索クエリ
DEFAULT_QUERIES = [
    "ペットボトル リサイクル",
    "ペットボトル 自動販売機",
    "コンビニ 飲み物 陳列",
    "ペットボトル お茶 紹介",
    "ペットボトル 工場 製造",
    "スーパー 飲料 棚",
    "水 ペットボトル レビュー",
    "ペットボトル ゴミ 分別",
]
CC_FILTER = "license ~= '(?i)creative commons'"
# YouTube 検索の「Creative Commons」絞り込みフィルタ（sp パラメータ）。
# これを付けると検索結果が CC ライセンス動画のみになる（ytsearch では CC を拾えない）。
CC_SEARCH_SP = "EgIwAQ%3D%3D"


def cc_search_url(query):
    return ("https://www.youtube.com/results?search_query="
            + urllib.parse.quote(query) + "&sp=" + CC_SEARCH_SP)

POLY_EPS = 0.0015
MIN_POLY_PTS = 3


def hf_login():
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        for parent in [HERE, *HERE.parents]:
            env = parent / ".env"
            if env.exists():
                for line in env.read_text(encoding="utf-8").splitlines():
                    if line.strip().startswith("HF_TOKEN="):
                        token = line.split("=", 1)[1].strip().strip("\"'"); break
            if token:
                break
    if token:
        os.environ["HF_TOKEN"] = token
        try:
            from huggingface_hub import login
            login(token=token, add_to_git_credential=False)
        except Exception as e:
            print(f"[hf] login warning: {e!r}")


def mask_to_polys(mask):
    import cv2
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = mask.shape[:2]
    eps = max(1.0, math.sqrt(h * h + w * w) * POLY_EPS)
    out = []
    for c in cnts:
        if len(c) < MIN_POLY_PTS:
            continue
        ap = cv2.approxPolyDP(c, eps, True).reshape(-1, 2).astype(float)
        if len(ap) >= MIN_POLY_PTS:
            out.append(ap)
    return out


# ---------------------------------------------------------------------------
# 1) ダウンロード
# ---------------------------------------------------------------------------
def load_queries(args):
    """--queries > --queries-file > DEFAULT_QUERIES。--num-shards 指定時は行を分割。"""
    if args.queries:
        qs = list(args.queries)
    elif getattr(args, "queries_file", None):
        qs = []
        for line in Path(args.queries_file).read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if s and not s.startswith("#"):
                qs.append(s)
    else:
        qs = list(DEFAULT_QUERIES)
    n = getattr(args, "num_shards", 1) or 1
    k = getattr(args, "shard", 0) or 0
    if n > 1:
        qs = qs[k::n]            # ラウンドロビンで pod に分配
    return qs


def cmd_download(args):
    VID_DIR.mkdir(parents=True, exist_ok=True)
    have = {p.stem for p in VID_DIR.glob("*.mp4")}
    queries = load_queries(args)
    unlimited = args.max_videos is None or args.max_videos <= 0
    tgt = "∞" if unlimited else args.max_videos
    print(f"[dl] shard {getattr(args,'shard',0)}/{getattr(args,'num_shards',1)} "
          f"queries={len(queries)} 既存 {len(have)} 本 / 目標 {tgt} 本")
    for q in queries:
        if not unlimited and len(have) >= args.max_videos:
            break
        print(f"[dl] query='{q}' (CC filtered, end={args.search_per_query})")
        cmd = [
            "yt-dlp", cc_search_url(q),
            "--playlist-end", str(args.search_per_query),
            "--match-filters", CC_FILTER,          # 念のため二重チェック
            "--ignore-errors", "--no-warnings",
            "--download-archive", str(VID_DIR / "archive.txt"),  # 既DL/クエリ重複をスキップ
            "-f", "bv*[height<=720]+ba/b[height<=720]/b",
            "--merge-output-format", "mp4",
            "--max-filesize", f"{args.max_filesize_mb}M",
            "--write-info-json",
            # --- YouTube に負荷をかけないためのスロットリング ---
            "--sleep-requests", str(args.sleep_requests),   # メタデータ要求間スリープ
            "--sleep-interval", str(args.sleep_interval),   # DL前スリープ(下限)
            "--max-sleep-interval", str(args.max_sleep_interval),  # DL前スリープ(上限,ランダム)
            "--limit-rate", args.limit_rate,                # 1DLの帯域上限
            "--retries", "5", "--fragment-retries", "5",
            "--concurrent-fragments", "1",                  # 同時フラグメントDLを増やさない
            "-o", str(VID_DIR / "%(id)s.%(ext)s"),
        ]
        if not unlimited:
            cmd += ["--max-downloads", str(args.max_videos - len(have))]
        try:
            subprocess.run(cmd, check=False, timeout=args.dl_timeout)
        except subprocess.TimeoutExpired:
            print("  [dl] timeout, 次のクエリへ")
        have = {p.stem for p in VID_DIR.glob("*.mp4")}
        print(f"  -> 累計 {len(have)} 本")
    print(f"[dl] 完了: {len(have)} 本 -> {VID_DIR}")


# ---------------------------------------------------------------------------
# 2) 抽出 + SAM3 bottle 判定 + CLIP dedup
# ---------------------------------------------------------------------------
def load_sam3():
    import torch
    hf_login()
    if torch.cuda.is_available():
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_sam3_image_model(device=dev)
    return model, Sam3Processor, dev


def load_clip(dev):
    import open_clip, torch
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k", device=dev)
    model.eval()
    return model, preprocess, torch


def detect_bottles(proc_cls, model, dev, pil, score_th, min_area_frac):
    """全フレームに対し text 'bottle' 検出。score/面積で選別したマスク群を返す。"""
    proc = proc_cls(model, device=dev, confidence_threshold=0.2)
    st = proc.set_image(pil)
    st = proc.set_text_prompt(prompt="bottle", state=st)
    m = st.get("masks"); s = st.get("scores")
    if m is None or len(m) == 0:
        return []
    m = m.float().cpu().numpy(); s = s.float().cpu().numpy().reshape(-1)
    if m.ndim == 4:
        m = m[:, 0]
    W, H = pil.size
    frame_area = W * H
    out = []
    for i in range(len(m)):
        if s[i] < score_th:
            continue
        mk = m[i] > 0.5
        area = int(mk.sum())
        if area < min_area_frac * frame_area:
            continue
        out.append((mk, float(s[i])))
    return out


def cmd_process(args):
    import cv2
    emit = getattr(args, "emit_dir", None)
    if emit:
        emit = Path(emit)
        (emit / "frames").mkdir(parents=True, exist_ok=True)
    else:
        IMG_DIR.mkdir(parents=True, exist_ok=True)
        ANN_DIR.mkdir(parents=True, exist_ok=True)
        META_DIR.mkdir(parents=True, exist_ok=True)
        QA_DIR.mkdir(parents=True, exist_ok=True)

    vids = sorted(VID_DIR.glob("*.mp4"))
    if not vids:
        print("[process] 動画がありません。先に download を実行してください。"); return
    print(f"[process] 動画 {len(vids)} 本")

    model, proc_cls, dev = load_sam3()
    clip_model, clip_pre, torch = load_clip(dev)

    kept_embs = []          # 多様性判定用の正規化済み埋め込み
    images, annots, manifest_rows, previews = [], [], [], []
    emit_records = []       # emit モード: 埋め込み付きポータブルレコード
    img_id = 0; ann_id = 0
    n_seen = n_bottle = 0
    frame_cap = None if (args.max_frames is None or args.max_frames <= 0) else args.max_frames

    def clip_emb(pil):
        with torch.no_grad():
            x = clip_pre(pil).unsqueeze(0).to(dev)
            f = clip_model.encode_image(x)
            f = f / f.norm(dim=-1, keepdim=True)
        return f.float().cpu().numpy().reshape(-1)

    for vp in vids:
        if frame_cap is not None and len(images) >= frame_cap:
            break
        vid = vp.stem
        info = {}
        ij = vp.with_suffix(".info.json")
        if ij.exists():
            try:
                info = json.load(open(ij, encoding="utf-8"))
            except Exception:
                pass
        cap = cv2.VideoCapture(str(vp))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        step = max(1, int(round(fps / args.fps)))
        kept_this = 0
        fidx = 0
        while True:
            if frame_cap is not None and len(images) >= frame_cap:
                break
            ok = cap.grab()
            if not ok:
                break
            if fidx % step == 0:
                ok, frame = cap.retrieve()
                if ok and frame is not None:
                    n_seen += 1
                    pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    if max(pil.size) > args.det_size:
                        r = args.det_size / max(pil.size)
                        pil = pil.resize((int(pil.size[0] * r), int(pil.size[1] * r)))
                    bottles = detect_bottles(proc_cls, model, dev, pil,
                                             args.bottle_score, args.min_bottle_area)
                    if len(bottles) >= args.min_bottles:
                        n_bottle += 1
                        emb = clip_emb(pil)
                        sim = max((float(emb @ e) for e in kept_embs), default=0.0)
                        if sim < args.dedup_thresh:
                            kept_embs.append(emb)
                            # 検出は縮小画像座標。元フレーム座標へ戻して保存（元解像度で保存）
                            sx = frame.shape[1] / pil.size[0]
                            sy = frame.shape[0] / pil.size[1]
                            base = f"youtube_{vid}_{fidx}.jpg"
                            if emit:
                                fname_rel = base
                                out_path = emit / "frames" / base
                            else:
                                fname_rel = f"images/all/{base}"
                                out_path = DATASET_ROOT / fname_rel
                            cv2.imwrite(str(out_path), frame)
                            cur_anns = []
                            images.append({"id": img_id, "file_name": fname_rel,
                                           "width": frame.shape[1], "height": frame.shape[0]})
                            pv_masks = []
                            for mk, sc in bottles:
                                polys = []
                                for poly in mask_to_polys(mk):
                                    poly[:, 0] *= sx; poly[:, 1] *= sy
                                    flat = poly.reshape(-1).tolist()
                                    if len(flat) >= 6:
                                        polys.append(flat)
                                if not polys:
                                    continue
                                xs = np.concatenate([np.array(p[0::2]) for p in polys])
                                ys = np.concatenate([np.array(p[1::2]) for p in polys])
                                bx = [float(xs.min()), float(ys.min()),
                                      float(xs.max() - xs.min()), float(ys.max() - ys.min())]
                                ann = {"id": ann_id, "image_id": img_id, "category_id": 1,
                                       "segmentation": polys, "iscrowd": 0,
                                       "area": float(bx[2] * bx[3]),
                                       "bbox": bx, "seg_source": "sam3", "score": round(sc, 3)}
                                annots.append(ann); cur_anns.append(ann)
                                ann_id += 1
                                pv_masks.append((mk, bx, sx, sy))
                            mrow = {
                                "source": "youtube", "source_image_id": f"{vid}_{fidx}",
                                "file_name": fname_rel,
                                "num_annotations": len(pv_masks),
                                "original_file_name": f"{vid}.mp4",
                                "url": info.get("webpage_url", f"https://youtu.be/{vid}"),
                                "title": info.get("title", ""),
                                "license": info.get("license", ""),
                                "uploader": info.get("uploader", ""),
                            }
                            manifest_rows.append(mrow)
                            if emit:
                                emit_records.append({
                                    "file_name": base, "width": frame.shape[1],
                                    "height": frame.shape[0], "embedding": emb.tolist(),
                                    "annotations": cur_anns, "manifest": mrow})
                            if not emit and len(previews) < 40 and pv_masks:
                                previews.append((out_path, frame.copy(), pv_masks))
                            img_id += 1
                            kept_this += 1
            fidx += 1
        cap.release()
        print(f"  {vid}: kept {kept_this}  (累計 {len(images)}/{frame_cap if frame_cap else '∞'})")

    print(f"[process] 走査フレーム {n_seen} / ボトル有 {n_bottle} / 採用 {len(images)} "
          f"(dedup閾値 {args.dedup_thresh})")

    # emit モード（fleet/pod）: 埋め込み付きポータブル parts.json のみ出力（dataset は触らない）
    if emit:
        shard = getattr(args, "shard", 0) or 0
        out_parts = emit / f"parts_{shard}.json"
        json.dump({"records": emit_records}, open(out_parts, "w", encoding="utf-8"),
                  ensure_ascii=False)
        print(f"[emit] {out_parts}  records={len(emit_records)} frames -> {emit/'frames'}")
        return

    # COCO ステージング書き出し
    coco = {"info": {"description": "youtube CC pet-bottle frames"},
            "licenses": [], "images": images, "annotations": annots,
            "categories": [{"id": 1, "name": "bottle", "supercategory": "bottle"}]}
    out_ann = ANN_DIR / "instances_youtube_sam3merge.json"
    json.dump(coco, open(out_ann, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"[write] {out_ann}  images={len(images)} annotations={len(annots)}")

    mpath = META_DIR / "manifest_youtube.csv"
    if manifest_rows:
        with open(mpath, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
            w.writeheader(); w.writerows(manifest_rows)
        print(f"[write] {mpath}")

    # QA プレビュー
    for idx, (path, frame, pv) in enumerate(previews):
        ov = frame.copy()
        for mk, bx, sx, sy in pv:
            mk_full = cv2.resize(mk.astype(np.uint8), (frame.shape[1], frame.shape[0]),
                                 interpolation=cv2.INTER_NEAREST).astype(bool)
            ov[mk_full] = (0.5 * ov[mk_full] + np.array([0, 200, 0])).clip(0, 255).astype(np.uint8)
            x, y, w, h = map(int, bx)
            cv2.rectangle(ov, (x, y), (x + w, y + h), (255, 255, 255), 2)
        cv2.imwrite(str(QA_DIR / f"preview_{idx:03d}_{path.stem}.jpg"), ov)
    if previews:
        print(f"[qa] previews -> {QA_DIR}")


def cmd_all(args):
    cmd_download(args)
    cmd_process(args)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("download", "process", "all"):
        p = sub.add_parser(name)
        p.add_argument("--queries", nargs="*", default=None)
        p.add_argument("--queries-file", default=None, help="1行1クエリのファイル（# はコメント）")
        p.add_argument("--shard", type=int, default=0, help="このpodのシャード番号")
        p.add_argument("--num-shards", type=int, default=1, help="総シャード数（クエリをラウンドロビン分割）")
        p.add_argument("--emit-dir", default=None, help="fleet出力先（frames/ と parts_<shard>.json）")
        p.add_argument("--max-videos", type=int, default=30)
        p.add_argument("--search-per-query", type=int, default=12)
        p.add_argument("--max-filesize-mb", type=int, default=300)
        p.add_argument("--dl-timeout", type=int, default=1800)
        # YouTube への負荷を抑えるスロットリング（10台並列でも穏やかに）
        p.add_argument("--sleep-requests", type=float, default=1.5,
                       help="メタデータ要求の間隔秒")
        p.add_argument("--sleep-interval", type=float, default=3.0,
                       help="各DL前の最小スリープ秒")
        p.add_argument("--max-sleep-interval", type=float, default=8.0,
                       help="各DL前の最大スリープ秒（min〜max のランダム）")
        p.add_argument("--limit-rate", default="3M", help="1DLあたりの帯域上限（例 3M）")
        p.add_argument("--fps", type=float, default=0.5, help="抽出fps（0.5=2秒に1枚）")
        p.add_argument("--max-frames", type=int, default=500)
        p.add_argument("--det-size", type=int, default=1024, help="検出時の長辺上限")
        p.add_argument("--bottle-score", type=float, default=0.45)
        p.add_argument("--min-bottles", type=int, default=1)
        p.add_argument("--min-bottle-area", type=float, default=0.0008, help="フレーム面積比の下限")
        p.add_argument("--dedup-thresh", type=float, default=0.88, help="CLIPコサイン類似がこれ以上なら捨てる")
    args = ap.parse_args()
    {"download": cmd_download, "process": cmd_process, "all": cmd_all}[args.cmd](args)


if __name__ == "__main__":
    main()
