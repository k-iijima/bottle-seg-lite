#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""bottle の COCO データセットを CVAT インポート用 zip に変換する。

VSLAM の segmentation/export_coco_for_cvat.py を、本リポジトリのレイアウトに合わせて
調整したもの。差分:
  - 画像は split ごとのフォルダではなく **images/all/** に一括配置（file_name は
    "images/all/<name>.jpg"）。basename 化すれば CVAT 用にできる。
  - 読み込むアノテは instances_<split><SUFFIX>.json（既定 SUFFIX="_sam3full"）。
  - basename は source 接頭辞付き（例 coco_train2017_15239.jpg）で全体一意なので衝突しない。
  - --only-source で「機械生成ラベルを含む画像だけ」に絞った軽量レビュータスクを作れる。
  - --image-list <txt> で検品対象の basename 一覧に絞り、--subset で CVAT 上のタスク名を
    付けた検品セットを作れる（例: qa_sam3full のレポートから作った multi-cap 検品セット）。

CVAT 取り込みの肝（datumaro COCO importer 準拠）:
  - file_name は **basename のみ**（images/ や subset/ を含めるとパス二重で読めない）。
  - zip 構造は images/<subset>/<name>, annotations/instances_<subset>.json。
  - annotation["attributes"] はラベル定義（cvat_labels_parts.json）に同名属性があれば取り込まれる。

使い方:
  python export_coco_for_cvat.py --splits val test
  python export_coco_for_cvat.py --splits all --subset review_multicap \
      --image-list datasets/bottle/qa_sam3full/review_multicap_images.txt
（パスに非ASCIIを含む環境で問題が出る場合は seg コンテナ内で実行:
  docker compose --profile tools run --rm seg python export_coco_for_cvat.py ...）
"""
from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATASET_ROOT = HERE / "datasets" / "bottle"
CVAT_IMAGE_DIR = "images"


def basename_of(file_name: str) -> str:
    return file_name.replace("\\", "/").rsplit("/", 1)[-1]


def build_cvat_zip(dataset_root: Path, split: str, ann_path: Path,
                   out_zip: Path, only_source: set[str] | None,
                   image_list: set[str] | None = None) -> tuple[int, int]:
    if not ann_path.exists():
        raise FileNotFoundError(f"アノテーション JSON がありません: {ann_path}")
    coco = json.loads(ann_path.read_text(encoding="utf-8"))

    anns_by_img: dict[int, list] = {}
    for a in coco.get("annotations", []):
        anns_by_img.setdefault(a["image_id"], []).append(a)

    # 画像を選別（only_source 指定時はその seg_source を含む画像だけ）
    images = coco.get("images", [])
    if only_source:
        images = [im for im in images
                  if any(a.get("seg_source") in only_source
                         for a in anns_by_img.get(im["id"], []))]
    if image_list is not None:
        images = [im for im in images if basename_of(im["file_name"]) in image_list]
        missing = image_list - {basename_of(im["file_name"]) for im in images}
        if missing:
            print(f"  [warn] image-list のうち {len(missing)} 件がアノテ内に見つからない")
    keep_ids = {im["id"] for im in images}

    # file_name を basename 化。本データは TACO 由来で同名画像が二重登録されている
    # （同一ファイルを別 image_id で重複登録, 約263種）。CVAT は同一タスク内で同名画像を
    # 持てないため、衝突時は zip 内名を __id<image_id> 付きで一意化する（無損失）。
    # src_base=実体探索用の元 basename, zip_name=zip 内＆file_name 用の一意名。
    src_base_of: dict[int, str] = {}
    zip_name_of: dict[int, str] = {}
    used: set[str] = set()
    n_collision = 0
    new_images = []
    for im in images:
        src_base = basename_of(im["file_name"])
        zip_name = src_base
        if zip_name in used:
            stem, _, ext = src_base.rpartition(".")
            zip_name = f"{stem}__id{im['id']}.{ext}" if ext else f"{src_base}__id{im['id']}"
            n_collision += 1
        used.add(zip_name)
        src_base_of[im["id"]] = src_base
        zip_name_of[im["id"]] = zip_name
        ni = dict(im)
        ni["file_name"] = zip_name
        new_images.append(ni)
    if n_collision:
        print(f"  [info] 同名画像を一意化: {n_collision} 件 (__id 付与, 主に TACO 重複)")

    # datumaro(CVAT) は全アノテに iscrowd を要求するが、元データ(COCO/LVIS/TACO)由来は
    # 一部欠落している。export 時に 0 で補完する（source JSON は変更しない）。area も念のため補完。
    kept_anns = []
    n_fix = 0
    for a in coco.get("annotations", []):
        if a["image_id"] not in keep_ids:
            continue
        if "iscrowd" not in a:
            a = {**a, "iscrowd": 0}
            n_fix += 1
        else:
            a = {**a, "iscrowd": int(a["iscrowd"])}
        if "area" not in a:
            x, y, w, h = a.get("bbox", [0, 0, 0, 0])
            a["area"] = float(w) * float(h)
        kept_anns.append(a)
    if n_fix:
        print(f"  [info] iscrowd 欠落を補完: {n_fix} 件")

    new_coco = dict(coco)
    new_coco["images"] = new_images
    new_coco["annotations"] = kept_anns

    src_img_dir = dataset_root / "images" / "all"
    subset = out_zip.stem.replace("_cvat_coco", "")
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for im in new_images:
            src = src_img_dir / src_base_of[im["id"]]
            if not src.exists():
                raise FileNotFoundError(f"画像実体が見つかりません: {src}")
            zf.write(src, f"{CVAT_IMAGE_DIR}/{subset}/{im['file_name']}")
        zf.writestr(f"annotations/instances_{subset}.json",
                    json.dumps(new_coco, ensure_ascii=False))
    return len(new_images), len(new_coco["annotations"])


def main(argv=None):
    ap = argparse.ArgumentParser(description="bottle COCO -> CVAT インポート zip")
    ap.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    ap.add_argument("--suffix", default="_sam3full",
                    help="読むアノテ: instances_<split><suffix>.json（既定 _sam3full）")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="zip 出力先（既定 <dataset-root>/cvat）")
    ap.add_argument("--only-source", nargs="*", default=None,
                    help="この seg_source を含む画像だけに絞る（例: sam3_text sam3）")
    ap.add_argument("--image-list", type=Path, default=None,
                    help="basename 一覧 txt（1行1ファイル名）。この画像だけに絞る")
    ap.add_argument("--subset", default=None,
                    help="出力 zip / CVAT タスク名（既定は split 名）")
    args = ap.parse_args(argv)

    out_dir = args.out_dir or (args.dataset_root / "cvat")
    only = set(args.only_source) if args.only_source else None
    image_list = None
    if args.image_list:
        image_list = {ln.strip().rsplit("/", 1)[-1]
                      for ln in args.image_list.read_text(encoding="utf-8").splitlines()
                      if ln.strip()}
        print(f"[image-list] {len(image_list)} 件: {args.image_list}")
    for split in args.splits:
        ann = args.dataset_root / "annotations" / f"instances_{split}{args.suffix}.json"
        tag = "_" + "_".join(sorted(only)) if only else ""
        name = args.subset or f"{split}{tag}"
        out_zip = out_dir / f"{name}_cvat_coco.zip"
        n_img, n_ann = build_cvat_zip(args.dataset_root, split, ann, out_zip, only, image_list)
        print(f"[OK][{split}] {out_zip}  images={n_img} annotations={n_ann}")
    print("[CVAT] Project 作成時に cvat_labels_parts.json を Raw に貼付（bottle/cap/label + 10属性）。"
          "Actions → Import dataset → 形式 COCO 1.0 でこの zip を指定。"
          "修正後は Export dataset(COCO 1.0)で書き出し。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
