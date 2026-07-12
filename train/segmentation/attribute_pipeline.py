#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ボトル属性の自動アノテ（認識モデル用の下書き）。ローカルでも RunPod でも動く。
属性は ALL_ATTRS の10種（material / cap / cap_color / label / label_color / fill_level /
crushed / visibility / orientation / depiction(実物か絵・印刷か)）、各 unknown あり。

  - material backend を選べる:
      --material-backend vlm  : 強い VLM に材質も答えさせる（精度重視・推奨）
      --material-backend clip : CLIP(open_clip ViT-H-14) ゼロショット（速い）
  - cap/label は常に VLM(VQA)。
  - VRAM とモデルサイズを見て dtype/バッチ/4bit を自動決定（大モデルは自動で 4bit）。

出力: instances_*_sam3attr.json（attributes と attr_conf を付与）。元 _sam3merge は無変更。

RunPod 推奨（精度重視）:
  python attribute_pipeline.py --data-root ./bottle \
      --vlm-model Qwen/Qwen3-VL-30B-A3B-Instruct --material-backend vlm --vlm-min 96
"""
from __future__ import annotations
import argparse, json, os, gc, re
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image
from tqdm.auto import tqdm

HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = HERE / "datasets" / "bottle"
SRC_ALL = "instances_all_sam3merge.json"
SRC_SPLITS = ["instances_train_sam3merge.json", "instances_val_sam3merge.json", "instances_test_sam3merge.json"]
SUF_FROM, SUF_TO = "_sam3merge", "_sam3attr"

CLIP_MATERIAL = [
    ("pet", "a transparent plastic PET drink bottle"), ("glass", "a glass bottle"),
    ("can", "an aluminum metal can"), ("other", "some other container or object"),
]

def _color_map(colors):
    m = {c: c for c in colors}
    m.update({"grey": "silver" if "silver" in colors else "black", "clear": "transparent",
              "colourless": "transparent", "see-through": "transparent", "none": "none", "no": "none"})
    return {k: v for k, v in m.items() if v in colors or v == "none"}

LABEL_COLORS = ["white", "black", "red", "blue", "green", "yellow", "orange", "brown", "multicolor"]
CAP_COLORS = ["white", "black", "red", "blue", "green", "yellow", "orange", "silver", "transparent"]

# VLM の質問と、答え(1語) -> 属性値 の対応。mapping は「先に一致した key」を採用するので
# 曖昧な場合に優先したい値を前に置く（例: visibility は occluded 系を先に）。
VLM_QUESTIONS = {
    "material": ("What is this container mainly made of? Answer one word: plastic, glass, metal, or other.",
                 {"plastic": "pet", "pet": "pet", "glass": "glass", "metal": "can", "can": "can",
                  "aluminum": "can", "aluminium": "can", "other": "other"}),
    "cap": ("Look only at the very top of this bottle. Is it closed with a cap or lid? "
            "Answer one word: yes, no, or unsure.", {"yes": "capped", "no": "uncapped"}),
    "cap_color": ("What is the color of this bottle's cap or lid? If there is no cap, answer none. "
                  "Answer one color word.", _color_map(CAP_COLORS)),
    "label": ("Does the body of this bottle have a printed or paper label on it? "
              "Answer one word: yes, no, or unsure.", {"yes": "labeled", "no": "unlabeled"}),
    "label_color": ("What is the dominant color of this bottle's label? If there is no label, answer none. "
                    "Answer one color word.", _color_map(LABEL_COLORS)),
    "fill_level": ("If the liquid level inside is visible, how full is this bottle? "
                   "Answer one word: empty, low, half, high, full, or unknown.",
                   {"empty": "empty", "low": "low", "half": "half", "high": "high", "full": "full",
                    "unknown": "unknown"}),
    "crushed": ("Is this bottle crushed, dented or deformed, or is it intact? "
                "Answer one word: crushed or intact.",
                {"crushed": "crushed", "dented": "crushed", "deformed": "crushed", "squashed": "crushed",
                 "intact": "intact", "normal": "intact", "no": "intact"}),
    "visibility": ("Is the whole bottle fully visible, or is part of it cut off or hidden behind something? "
                   "Answer one word: full or occluded.",
                   {"occluded": "occluded", "hidden": "occluded", "cut": "occluded", "partial": "occluded",
                    "behind": "occluded", "full": "full", "fully": "full", "visible": "full"}),
    "orientation": ("Is this bottle standing upright or lying down on its side? "
                    "Answer one word: upright or lying.",
                    {"lying": "lying", "sideways": "lying", "horizontal": "lying", "side": "lying",
                     "upright": "upright", "standing": "upright", "vertical": "upright"}),
    "depiction": ("Is this bottle a real physical object photographed by a camera, or a depiction "
                  "such as a drawing, cartoon, illustration, 3D render, or an image printed on "
                  "paper or shown on a screen? Answer one word: real or depicted.",
                  {"depicted": "depicted", "drawn": "depicted", "drawing": "depicted",
                   "cartoon": "depicted", "illustration": "depicted", "render": "depicted",
                   "printed": "depicted", "screen": "depicted", "painting": "depicted",
                   "graphic": "depicted", "real": "real", "physical": "real", "photo": "real"}),
}
ALL_ATTRS = ["material", "cap", "cap_color", "label", "label_color",
             "fill_level", "crushed", "visibility", "orientation", "depiction"]
MAT_MARGIN = 0.10


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
            from huggingface_hub import login
            login(token=token, add_to_git_credential=False)
        except Exception:
            pass


def est_params_b(model_id):
    m = re.findall(r"(\d+)B", model_id)        # 8B / 32B / 30B-A3B(->30,3) / 235B
    return int(m[0]) if m else 8


def auto_config(torch, model_id):
    if not torch.cuda.is_available():
        return dict(device="cpu", dtype=torch.float32, quant=False, vlm_batch=1, clip_batch=32, vram=0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    est = est_params_b(model_id)
    quant = (est * 2 * 1.3) > (vram * 0.9) or vram < 18      # bf16 で収まらなければ 4bit
    vb = 24 if vram >= 70 else 16 if vram >= 40 else 8 if vram >= 20 else 4 if vram >= 12 else 2
    if est >= 100:
        vb = max(1, vb // 8)
    elif est >= 28:
        vb = max(1, vb // 3)
    elif est >= 14:
        vb = max(1, vb // 2)
    return dict(device="cuda", dtype=dtype, quant=quant, vlm_batch=vb, clip_batch=64, vram=vram, est=est)


def iter_crops(targets, imgs, root, pad=0.12, min_up=0):
    by_img = {}
    for a in targets:
        by_img.setdefault(a["image_id"], []).append(a)
    for iid, alist in by_img.items():
        try:
            pil = Image.open(root / imgs[iid]["file_name"]).convert("RGB")
        except Exception:
            continue
        W, H = pil.size
        for a in alist:
            x, y, w, h = a["bbox"]; p = pad * max(w, h)
            box = (max(0, int(x - p)), max(0, int(y - p)), min(W, int(x + w + p)), min(H, int(y + h + p)))
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            crop = pil.crop(box)
            if min_up and max(crop.size) < min_up:
                s = min_up / max(crop.size)
                crop = crop.resize((int(crop.size[0] * s), int(crop.size[1] * s)))
            yield a, crop


def run_material_clip(targets, imgs, root, cfg, model_name="ViT-H-14", pretrained="laion2b_s32b_b79k"):
    import torch, open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=cfg["device"])
    model.eval()
    tok = open_clip.get_tokenizer(model_name)
    with torch.no_grad():
        tf = model.encode_text(tok([p for _, p in CLIP_MATERIAL]).to(cfg["device"]))
        tf /= tf.norm(dim=-1, keepdim=True)
    print(f"[material] CLIP {model_name}/{pretrained}")
    res = {}; bt, ba = [], []

    def flush():
        if not bt:
            return
        with torch.no_grad():
            f = model.encode_image(torch.stack(bt).to(cfg["device"])); f /= f.norm(dim=-1, keepdim=True)
            probs = (100.0 * f @ tf.T).softmax(-1).float().cpu().numpy()
        for i, a in enumerate(ba):
            o = probs[i].argsort()[::-1]
            val = CLIP_MATERIAL[o[0]][0] if probs[i][o[0]] - probs[i][o[1]] >= MAT_MARGIN else "unknown"
            res[a["id"]] = (val, round(float(probs[i][o[0]]), 3))
        bt.clear(); ba.clear()

    for a, crop in tqdm(iter_crops(targets, imgs, root), total=len(targets), desc="material(clip)"):
        bt.append(preprocess(crop)); ba.append(a)
        if len(bt) >= cfg["clip_batch"]:
            flush()
    flush()
    del model; gc.collect()
    if cfg["device"] == "cuda":
        torch.cuda.empty_cache()
    return res


def run_vlm(targets, imgs, root, cfg, attrs, model_id):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from qwen_vl_utils import process_vision_info
    kw = dict(device_map=cfg["device"], dtype=cfg["dtype"])
    if cfg["quant"]:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=cfg["dtype"], bnb_4bit_use_double_quant=True)
    model = AutoModelForImageTextToText.from_pretrained(model_id, **kw).eval()
    proc = AutoProcessor.from_pretrained(model_id, min_pixels=200 * 200, max_pixels=640 * 640)
    proc.tokenizer.padding_side = "left"
    print(f"[vlm] {model_id} attrs={attrs} quant4bit={cfg['quant']} batch={cfg['vlm_batch']}")

    def ask(pils, q):
        msgs = [[{"role": "user", "content": [{"type": "image", "image": p}, {"type": "text", "text": q}]}] for p in pils]
        texts = [proc.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]
        ims, _ = process_vision_info(msgs)
        inp = proc(text=texts, images=ims, padding=True, return_tensors="pt").to(cfg["device"])
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=6, do_sample=False)
        return [s.strip().lower() for s in proc.batch_decode(out[:, inp.input_ids.shape[1]:], skip_special_tokens=True)]

    res = {}; ba, bp = [], []

    def flush():
        if not bp:
            return
        per = {}
        for attr in attrs:
            q, mapping = VLM_QUESTIONS[attr]
            ans = ask(bp, q)
            vals = []
            for s in ans:
                v = "unknown"
                for k, mapped in mapping.items():
                    if k in s:
                        v = mapped; break
                vals.append(v)
            per[attr] = vals
        for i, a in enumerate(ba):
            res[a["id"]] = {attr: per[attr][i] for attr in attrs}
        ba.clear(); bp.clear()

    for a, crop in tqdm(iter_crops(targets, imgs, root, min_up=256), total=len(targets), desc="vlm " + ",".join(attrs)):
        ba.append(a); bp.append(crop)
        if len(bp) >= cfg["vlm_batch"]:
            flush()
    flush()
    del model; gc.collect()
    if cfg["device"] == "cuda":
        torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--material-min", type=int, default=48)
    ap.add_argument("--vlm-min", type=int, default=64)
    ap.add_argument("--vlm-model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--material-backend", choices=["clip", "vlm"], default="clip")
    ap.add_argument("--attrs", default=",".join(ALL_ATTRS),
                    help="付与する属性(カンマ区切り)。既定は全9属性: " + ",".join(ALL_ATTRS))
    ap.add_argument("--limit-material", type=int, default=None)
    ap.add_argument("--limit-vlm", type=int, default=None)
    ap.add_argument("--num-shards", type=int, default=1, help="並列分散の総ワーカー数")
    ap.add_argument("--shard", type=int, default=0, help="このワーカーの番号 (0..num_shards-1)")
    ap.add_argument("--emit-attrs", default=None,
                    help="指定時: {ann_id: {attributes, attr_conf}} だけを JSON 出力（マージは merge_attrs.py）")
    args = ap.parse_args()
    attrs = [a.strip() for a in args.attrs.split(",") if a.strip()]
    bad = [a for a in attrs if a not in ALL_ATTRS]
    if bad:
        ap.error(f"未知の属性: {bad}. 使えるのは {ALL_ATTRS}")

    root = args.data_root; ann_dir = root / "annotations"
    coco = json.load(open(ann_dir / SRC_ALL, encoding="utf-8"))
    imgs = {i["id"]: i for i in coco["images"]}
    poly = [a for a in coco["annotations"] if not isinstance(a.get("segmentation"), dict)]

    hf_login()
    import torch
    cfg = auto_config(torch, args.vlm_model)
    print(f"[cfg] device={cfg['device']} vram={cfg.get('vram',0):.0f}GB est={cfg.get('est','?')}B "
          f"dtype={cfg['dtype']} quant4bit={cfg['quant']} vlm_batch={cfg['vlm_batch']}")

    mat_res, vlm_res = {}, {}
    clip_material = args.material_backend == "clip" and "material" in attrs
    # material backend = clip かつ material 要求時のみ CLIP を回す
    def shard_of(t):
        # 対象リストは JSON 順で決定的。全 pod が同じ num_shards/min 前提で分割する
        return t[args.shard::args.num_shards] if args.num_shards > 1 else t

    if clip_material:
        t = [a for a in poly if max(a["bbox"][2], a["bbox"][3]) >= args.material_min]
        if args.limit_material:
            t = t[: args.limit_material]
        t = shard_of(t)
        print(f"[material] clip targets {len(t)} (>= {args.material_min}px)")
        mat_res = run_material_clip(t, imgs, root, cfg)

    vlm_attrs = [a for a in attrs if not (a == "material" and clip_material)]
    if vlm_attrs:
        t = [a for a in poly if max(a["bbox"][2], a["bbox"][3]) >= args.vlm_min]
        if args.limit_vlm:
            t = t[: args.limit_vlm]
        t = shard_of(t)
        print(f"[vlm] targets {len(t)} (>= {args.vlm_min}px) attrs={vlm_attrs} "
              f"(~{len(t) * len(vlm_attrs)} generations)")
        vlm_res = run_vlm(t, imgs, root, cfg, vlm_attrs, args.vlm_model)

    def attrs_for(a):
        v = vlm_res.get(a["id"], {})
        conf = {}
        out = {}
        for attr in attrs:
            if attr == "material" and clip_material:
                mat, mconf = mat_res.get(a["id"], ("unknown", None))
                out["material"] = mat
                if mconf is not None:
                    conf["material_conf"] = mconf
            else:
                out[attr] = v.get(attr, "unknown")
        return out, conf

    # シャードモード: 処理した ann だけ {id: {attributes, attr_conf}} を出力して終了
    if args.emit_attrs:
        done_ids = set(mat_res) | set(vlm_res)
        emit = {}
        for a in poly:
            if a["id"] not in done_ids:
                continue
            av, conf = attrs_for(a)
            emit[str(a["id"])] = {"attributes": av, **({"attr_conf": conf} if conf else {})}
        json.dump(emit, open(args.emit_attrs, "w", encoding="utf-8"), ensure_ascii=False)
        print(f"[emit] {args.emit_attrs} ({len(emit)} anns)")
        return

    def write(src):
        d = json.load(open(ann_dir / src, encoding="utf-8"))
        for a in d["annotations"]:
            attrs, conf = attrs_for(a)
            a["attributes"] = attrs
            if conf:
                a["attr_conf"] = conf
        out = ann_dir / src.replace(SUF_FROM, SUF_TO)
        json.dump(d, open(out, "w", encoding="utf-8"), ensure_ascii=False)
        print("[write]", out)

    write(SRC_ALL)
    for s in SRC_SPLITS:
        if (ann_dir / s).exists():
            write(s)

    allids = set(mat_res) | set(vlm_res)
    def col(attr):
        c = Counter()
        for i in allids:
            v = vlm_res.get(i, {}).get(attr)
            if v is None and attr == "material" and clip_material:
                v = mat_res.get(i, ("unknown",))[0]
            c[v or "unknown"] += 1
        return dict(c)
    for attr in attrs:
        print(f"{attr}:", col(attr))


if __name__ == "__main__":
    main()
