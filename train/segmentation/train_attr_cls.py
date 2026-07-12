#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ボトル属性認識モデル(2段目の軽量マルチヘッド分類器)の学習一式。

  extract : _sam3full の属性付き bottle(長辺>=96px)を bbox+15% pad でクロップし
            attr_crops/{split}/*.jpg + {split}.jsonl を作る(ローカル実行、転送軽量化)
  train   : クロップからマルチヘッド分類器を学習(RunPod想定、torch/torchvisionのみ)。
            unknown はロス無視(ignore_index)、クラス重みは逆頻度sqrt、
            val の平均 macro-F1 でベスト選択 → test 評価 + ONNX 出力。

属性スキーマは DATASET.md §5 / attribute_pipeline.py の10種。教師は Qwen3-VL 疑似ラベル
なので、精度は「対疑似ラベル再現度」として扱うこと。

  python train_attr_cls.py extract --data-root datasets/bottle
  python train_attr_cls.py train --crops-root attr_crops --arch mobilenet_v3_small
"""
from __future__ import annotations
import argparse, json, math, os, sys, time
from pathlib import Path

# 値の順序は固定(ONNX の logits オフセットに直結するので変更禁止)
ATTR_CLASSES = {
    "material":    ["pet", "glass", "can", "other"],
    "cap":         ["capped", "uncapped"],
    "cap_color":   ["none", "white", "black", "blue", "red", "green", "yellow",
                    "silver", "orange", "transparent"],
    "label":       ["labeled", "unlabeled"],
    "label_color": ["none", "white", "blue", "green", "yellow", "red", "black",
                    "orange", "brown", "multicolor"],
    "fill_level":  ["empty", "low", "half", "high", "full"],
    "crushed":     ["intact", "crushed"],
    "visibility":  ["full", "occluded"],
    "orientation": ["upright", "lying"],
    "depiction":   ["real", "depicted"],
}
ATTRS = list(ATTR_CLASSES)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------- extract

def crop_one(img, bbox, pad=0.15):
    x, y, w, h = bbox
    px, py = w * pad, h * pad
    l = max(0, int(x - px)); t = max(0, int(y - py))
    r = min(img.width, int(x + w + px)); b = min(img.height, int(y + h + py))
    if r - l < 8 or b - t < 8:  # 画像外・退化 bbox はスキップ
        return None
    return img.crop((l, t, r, b))


def _extract_image(task):
    from PIL import Image
    img_path, out_dir, anns, long_side = task
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        return [("ERR", f"{img_path}: {e!r}")]
    rows = []
    for a in anns:
        c = crop_one(img, a["bbox"])
        if c is None:
            rows.append(("ERR", f"{img_path}: ann {a['id']} degenerate bbox {a['bbox']}"))
            continue
        if max(c.size) > long_side:
            s = long_side / max(c.size)
            c = c.resize((max(1, round(c.width * s)), max(1, round(c.height * s))),
                         Image.BILINEAR)
        fn = f"{a['id']}.jpg"
        c.save(out_dir / fn, quality=92)
        rows.append(("OK", {"f": fn, "a": a["attributes"]}))
    return rows


def cmd_extract(args):
    from concurrent.futures import ProcessPoolExecutor
    from tqdm import tqdm
    root = Path(args.data_root)
    out_root = Path(args.out)
    for split in ["train", "val", "test"]:
        d = json.load(open(root / "annotations" / f"instances_{split}_sam3full.json",
                           encoding="utf-8"))
        imgs = {im["id"]: im for im in d["images"]}
        by_img = {}
        n = 0
        for a in d["annotations"]:
            if a["category_id"] != 1 or a.get("iscrowd"):
                continue
            attrs = a.get("attributes") or {}
            if not any(v not in (None, "unknown") for v in attrs.values()):
                continue
            if max(a["bbox"][2], a["bbox"][3]) < args.min_side:
                continue
            by_img.setdefault(a["image_id"], []).append(a)
            n += 1
        out_dir = out_root / split
        out_dir.mkdir(parents=True, exist_ok=True)
        tasks = [(str(root / "images" / "all" / Path(imgs[i]["file_name"]).name),
                  out_dir, anns, args.long_side) for i, anns in by_img.items()]
        rows, errs = [], []
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for res in tqdm(ex.map(_extract_image, tasks, chunksize=16),
                            total=len(tasks), desc=split):
                for st, r in res:
                    (rows if st == "OK" else errs).append(r)
        with open(out_root / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[extract] {split}: {len(rows)}/{n} crops -> {out_dir}  (errors {len(errs)})")
        for e in errs[:5]:
            print("  ", e)


# ---------------------------------------------------------------- train

def load_split(crops_root, split):
    rows = []
    with open(Path(crops_root) / f"{split}.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            labels = []
            for k in ATTRS:
                v = (r["a"] or {}).get(k)
                cls = ATTR_CLASSES[k]
                labels.append(cls.index(v) if v in cls else -1)
            rows.append((f"{split}/{r['f']}", labels))
    return rows


def build_model(arch, pretrained=True):
    import torch.nn as nn
    import torchvision.models as tvm

    class MultiHead(nn.Module):
        def __init__(self):
            super().__init__()
            if arch == "mobilenet_v3_small":
                m = tvm.mobilenet_v3_small(
                    weights=tvm.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None)
            elif arch == "mobilenet_v3_large":
                m = tvm.mobilenet_v3_large(
                    weights=tvm.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None)
            elif arch.startswith("mobilenetv4_"):
                import timm  # V4 は torchvision 未収録(conv系のみ許可: hybrid は MQA が量子化に不利)
                m = timm.create_model(arch, pretrained=pretrained, num_classes=0)
                # timm の efficientnet 系は num_classes=0 でも conv_head 込みの
                # head_hidden_size 次元(例: v4_conv_small は 1280)が出てくる
                self.backbone, in_dim = m, getattr(m, "head_hidden_size", 0) or m.num_features
            else:
                raise SystemExit(f"unknown arch {arch}")
            if not hasattr(self, "backbone"):
                self.backbone = nn.Sequential(m.features, m.avgpool, nn.Flatten(1))
                in_dim = m.classifier[0].in_features
            self.neck = nn.Sequential(nn.Linear(in_dim, 512), nn.Hardswish(),
                                      nn.Dropout(0.2))
            self.heads = nn.ModuleList(
                [nn.Linear(512, len(ATTR_CLASSES[k])) for k in ATTRS])

        def forward(self, x):
            x = self.neck(self.backbone(x))
            return [h(x) for h in self.heads]

    return MultiHead()


class CropDataset:
    def __init__(self, crops_root, rows, img_size, train):
        self.root, self.rows, self.size, self.train = Path(crops_root), rows, img_size, train

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        import torch
        from PIL import Image
        import random
        fn, labels = self.rows[i]
        img = Image.open(self.root / fn).convert("RGB")
        if self.train:
            # 控えめな RandomResizedCrop 相当(強く切ると cap/fill が消える)
            s = random.uniform(0.8, 1.0)
            w, h = img.size
            cw, ch = round(w * s), round(h * s)
            x = random.randint(0, w - cw); y = random.randint(0, h - ch)
            img = img.crop((x, y, x + cw, y + ch))
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
        img = img.resize((self.size, self.size), Image.BILINEAR)  # アスペクトは潰す(ランタイムと同じ規約)
        t = torch.from_numpy(__import__("numpy").asarray(img).copy()).permute(2, 0, 1).float() / 255
        if self.train:
            b = random.uniform(0.8, 1.2); c = random.uniform(0.8, 1.2)
            t = ((t - 0.5) * c + 0.5) * b  # 色相はいじらない(色属性があるため)
            t = t.clamp(0, 1)
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        return (t - mean) / std, torch.tensor(labels, dtype=torch.long)


def class_weights(rows):
    import torch
    ws = []
    for ai, k in enumerate(ATTRS):
        n_cls = len(ATTR_CLASSES[k])
        cnt = [0] * n_cls
        for _, labels in rows:
            if labels[ai] >= 0:
                cnt[labels[ai]] += 1
        total = sum(cnt)
        w = [min(4.0, max(0.25, math.sqrt(total / (n_cls * c)))) if c > 0 else 1.0
             for c in cnt]
        ws.append(torch.tensor(w))
    return ws


def evaluate(model, loader, device):
    """attr ごとの acc / macro-F1(support>0 のクラスのみ)を返す。"""
    import torch
    model.eval()
    K = [len(ATTR_CLASSES[k]) for k in ATTRS]
    tp = [torch.zeros(k) for k in K]; fp = [torch.zeros(k) for k in K]
    fn = [torch.zeros(k) for k in K]; sup = [torch.zeros(k) for k in K]
    correct = [0] * len(ATTRS); valid = [0] * len(ATTRS)
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            outs = model(x)
            for ai in range(len(ATTRS)):
                t = y[:, ai]
                m = t >= 0
                if not m.any():
                    continue
                p = outs[ai].argmax(1).cpu()[m]; t = t[m]
                correct[ai] += (p == t).sum().item(); valid[ai] += len(t)
                for c in range(K[ai]):
                    tp[ai][c] += ((p == c) & (t == c)).sum()
                    fp[ai][c] += ((p == c) & (t != c)).sum()
                    fn[ai][c] += ((p != c) & (t == c)).sum()
                    sup[ai][c] += (t == c).sum()
    res = {}
    for ai, k in enumerate(ATTRS):
        f1s = []
        for c in range(K[ai]):
            if sup[ai][c] == 0:
                continue
            prec = tp[ai][c] / max(1e-9, tp[ai][c] + fp[ai][c])
            rec = tp[ai][c] / max(1e-9, tp[ai][c] + fn[ai][c])
            f1s.append((2 * prec * rec / max(1e-9, prec + rec)).item())
        res[k] = {"acc": correct[ai] / max(1, valid[ai]),
                  "macro_f1": sum(f1s) / max(1, len(f1s)), "n": valid[ai],
                  "per_class_support": [int(s) for s in sup[ai]]}
    res["_mean_macro_f1"] = sum(res[k]["macro_f1"] for k in ATTRS) / len(ATTRS)
    return res


def export_onnx(model, img_size, out_path, device):
    """アプリと同じ流儀: uint8 NHWC RGBA [1,S,S,4] 入力を埋め込んだ ONNX を出す。"""
    import torch
    import torch.nn as nn

    class Wrap(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
            self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1) * 255)
            self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1) * 255)

        def forward(self, rgba):
            x = rgba[..., :3].to(torch.float32).permute(0, 3, 1, 2)
            x = (x - self.mean) / self.std
            return torch.cat(self.m(x), dim=1)

    w = Wrap(model).to(device).eval()
    dummy = torch.zeros(1, img_size, img_size, 4, dtype=torch.uint8, device=device)
    torch.onnx.export(w, dummy, str(out_path), opset_version=17,
                      input_names=["input"], output_names=["logits"])
    # 検算: ORT vs torch
    try:
        import numpy as np, onnxruntime as ort
        rng = np.random.default_rng(0)
        a = rng.integers(0, 256, (1, img_size, img_size, 4), dtype=np.uint8)
        sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
        o1 = sess.run(None, {"input": a})[0]
        with torch.no_grad():
            o2 = w(torch.from_numpy(a).to(device)).cpu().numpy()
        err = float(abs(o1 - o2).max())
        print(f"[onnx] max abs diff vs torch: {err:.5f}")
    except Exception as e:
        print("[onnx] verify skipped:", repr(e))


def cmd_train(args):
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    tr_rows = load_split(args.crops_root, "train")
    va_rows = load_split(args.crops_root, "val")
    te_rows = load_split(args.crops_root, "test")
    print(f"[data] train {len(tr_rows)} / val {len(va_rows)} / test {len(te_rows)}")
    ws = [w.to(device) for w in class_weights(tr_rows)]

    dl = dict(num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)
    tr = DataLoader(CropDataset(args.crops_root, tr_rows, args.img, True),
                    batch_size=args.batch, shuffle=True, drop_last=True, **dl)
    va = DataLoader(CropDataset(args.crops_root, va_rows, args.img, False),
                    batch_size=args.batch, **dl)
    te = DataLoader(CropDataset(args.crops_root, te_rows, args.img, False),
                    batch_size=args.batch, **dl)

    model = build_model(args.arch).to(device)
    if device == "cuda":
        model = model.to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
    steps = len(tr) * args.epochs
    warm = len(tr) * min(2, args.epochs)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / max(1, warm) if s < warm
        else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))

    best, hist = -1.0, []
    for ep in range(args.epochs):
        model.train()
        t0, tot, seen = time.time(), 0.0, 0
        for x, y in tr:
            x = x.to(device, non_blocking=True)
            if device == "cuda":
                x = x.to(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
                outs = model(x)
                losses = []
                for ai in range(len(ATTRS)):
                    t = y[:, ai]
                    if (t >= 0).any():
                        losses.append(F.cross_entropy(outs[ai].float(), t,
                                                      weight=ws[ai], ignore_index=-1))
                loss = torch.stack(losses).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step(); sched.step()
            tot += loss.item() * len(x); seen += len(x)
        m = evaluate(model, va, device)
        hist.append({"epoch": ep + 1, "train_loss": tot / max(1, seen), "val": m})
        print(f"[ep {ep+1:02d}/{args.epochs}] loss {tot/max(1,seen):.4f} "
              f"val meanF1 {m['_mean_macro_f1']:.4f} ({time.time()-t0:.0f}s) " +
              " ".join(f"{k}:{m[k]['macro_f1']:.2f}" for k in ATTRS))
        if m["_mean_macro_f1"] > best:
            best = m["_mean_macro_f1"]
            torch.save({"arch": args.arch, "img": args.img, "attrs": ATTR_CLASSES,
                        "state": model.state_dict()}, out / "best.pth")
    json.dump(hist, open(out / "history.json", "w"), indent=1)

    ck = torch.load(out / "best.pth", map_location=device, weights_only=False)
    model.load_state_dict(ck["state"])
    test_m = evaluate(model, te, device)
    print("[test] meanF1", f"{test_m['_mean_macro_f1']:.4f}")
    for k in ATTRS:
        print(f"  {k:12s} acc {test_m[k]['acc']:.3f} macroF1 {test_m[k]['macro_f1']:.3f} (n={test_m[k]['n']})")
    json.dump({"arch": args.arch, "val_best_mean_macro_f1": best, "test": test_m},
              open(out / "metrics.json", "w"), indent=1)

    export_onnx(model, args.img, out / f"attr_cls_{args.arch}.onnx", device)
    offs, o = {}, 0
    for k in ATTRS:
        offs[k] = {"offset": o, "classes": ATTR_CLASSES[k]}
        o += len(ATTR_CLASSES[k])
    json.dump({"input": f"uint8 [1,{args.img},{args.img},4] RGBA (NHWC)",
               "output": f"logits float32 [1,{o}]", "heads": offs},
              open(out / "onnx_meta.json", "w"), indent=1)
    print(f"[done] -> {out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--data-root", default="datasets/bottle")
    e.add_argument("--out", default="datasets/bottle/attr_crops")
    e.add_argument("--min-side", type=int, default=96)
    e.add_argument("--long-side", type=int, default=192)
    e.add_argument("--workers", type=int, default=8)
    t = sub.add_parser("train")
    t.add_argument("--crops-root", default="attr_crops")
    t.add_argument("--arch", default="mobilenet_v3_small",
                   choices=["mobilenet_v3_small", "mobilenet_v3_large",
                            "mobilenetv4_conv_small", "mobilenetv4_conv_medium"])
    t.add_argument("--out", default=None)
    t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--batch", type=int, default=256)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--img", type=int, default=128)
    t.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    if args.cmd == "extract":
        cmd_extract(args)
    else:
        if args.out is None:
            args.out = f"work_attr/{args.arch}"
        cmd_train(args)


if __name__ == "__main__":
    main()
