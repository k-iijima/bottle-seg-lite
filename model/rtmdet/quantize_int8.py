"""RTMDet-Ins ONNX を int8（QDQ・static）に量子化する。ホストの Python で実行:

  pip install onnx onnxruntime pillow numpy
  python model/rtmdet/quantize_int8.py

既定でデータセット画像から 128 枚をキャリブレーションに使い、
app/assets/models/rtmdet_ins_int8.onnx を出力、fp32 との出力比較も表示する。
"""
import argparse
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image

# detector.dart / export_rtmdet.sh と同一の前処理（BGR 順・mean/std 正規化・squash リサイズ）
MEAN_BGR = np.array([103.53, 116.28, 123.675], dtype=np.float32)
STD_BGR = np.array([57.375, 57.12, 58.395], dtype=np.float32)


def preprocess(path: Path, size: int) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    rgb = np.asarray(img, dtype=np.float32)
    bgr = rgb[:, :, ::-1]
    x = (bgr - MEAN_BGR) / STD_BGR
    return x.transpose(2, 0, 1)[None]  # NCHW


class ImageCalibReader:
    def __init__(self, files, size):
        self._iter = iter(files)
        self._size = size

    def get_next(self):
        path = next(self._iter, None)
        if path is None:
            return None
        return {"input": preprocess(path, self._size)}


def run_model(sess, x):
    return sess.run(None, {"input": x})


def summarize(name, outs, thr=0.4):
    dets, labels, masks = outs
    keep = dets[0][:, 4] >= thr
    print(f"  {name}: dets(score>={thr}) = {int(keep.sum())} "
          f"scores={np.round(dets[0][keep][:, 4], 3).tolist()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="app/assets/models/rtmdet_ins.onnx")
    ap.add_argument("--out", default="app/assets/models/rtmdet_ins_int8.onnx")
    ap.add_argument("--calib-dir",
                    default="train/segmentation/datasets/bottle/images/all")
    ap.add_argument("--num", type=int, default=128)
    ap.add_argument("--size", type=int, default=416)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import onnxruntime as ort
    from onnxruntime.quantization import (CalibrationMethod, QuantFormat,
                                          QuantType, quantize_static)
    from onnxruntime.quantization.shape_inference import quant_pre_process

    files = sorted(Path(args.calib_dir).glob("*.jpg"))
    random.Random(args.seed).shuffle(files)
    calib_files = files[: args.num]
    print(f"[calib] {len(calib_files)} images from {args.calib_dir}")

    pre = Path(args.out).with_suffix(".pre.onnx")
    print("[quant] preprocessing (shape inference)...")
    # skip_symbolic_shape=True: symbolic shape inference は torch を import するため
    # 回避（静的 shape モデルなので通常の shape inference で十分）
    quant_pre_process(args.model, str(pre), skip_symbolic_shape=True)

    # mmdeploy 出力は opset 11。per-channel 量子化（DequantizeLinear の axis 属性）
    # には opset 13+ が必要なので変換する
    import onnx
    from onnx import version_converter
    pre_model = onnx.load(str(pre))
    opset = {o.domain or "ai.onnx": o.version for o in pre_model.opset_import}
    if opset.get("ai.onnx", 0) < 13:
        print(f"[quant] converting opset {opset.get('ai.onnx')} -> 13...")
        pre_model = version_converter.convert_version(pre_model, 13)
        onnx.save(pre_model, str(pre))

    # SE-attention の fc（bias が initializer でない）と、活性値の外れ値が極端で
    # 単独量子化でも検出が全滅する stage2.1/blocks.0 の3 Conv を除外する
    # （単一 Conv ずつの bisect で特定。他の Conv は head 含め量子化しても精度維持）。
    model = onnx.load(str(pre))
    bad = ("attention", "/backbone/stage2/stage2.1/blocks/blocks.0/")
    exclude = [n.name for n in model.graph.node
               if any(b in n.name for b in bad)]
    print(f"[quant] static QDQ int8 quantization (exclude {len(exclude)} nodes)...")
    t0 = time.time()
    quantize_static(
        str(pre), args.out, ImageCalibReader(calib_files, args.size),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        calibrate_method=CalibrationMethod.MinMax,
        op_types_to_quantize=["Conv"],
        nodes_to_exclude=exclude,
    )
    pre.unlink(missing_ok=True)
    print(f"[quant] done in {time.time() - t0:.0f}s -> {args.out} "
          f"({Path(args.out).stat().st_size // 1_000_000} MB, "
          f"fp32: {Path(args.model).stat().st_size // 1_000_000} MB)")

    # --- fp32 と比較（検出数・スコア・CPU 実行時間） ---
    so = ort.SessionOptions()
    fp32 = ort.InferenceSession(args.model, so, providers=["CPUExecutionProvider"])
    int8 = ort.InferenceSession(args.out, so, providers=["CPUExecutionProvider"])
    test_files = files[args.num: args.num + 5]
    for f in test_files:
        x = preprocess(f, args.size)
        print(f"[check] {f.name}")
        summarize("fp32", run_model(fp32, x))
        summarize("int8", run_model(int8, x))
    x = preprocess(test_files[0], args.size)
    for name, sess in [("fp32", fp32), ("int8", int8)]:
        run_model(sess, x)  # warmup
        t0 = time.time()
        n = 5
        for _ in range(n):
            run_model(sess, x)
        print(f"[speed] {name}: {(time.time() - t0) / n * 1000:.0f} ms/frame (CPU)")


if __name__ == "__main__":
    main()
