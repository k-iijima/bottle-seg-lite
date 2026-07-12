"""前処理内蔵 ONNX を fp16 化する（WebGPU 向け。コンテナ内で実行）:

  docker compose --profile tools run --rm rtmdet-onnx \
    python /work/convert_fp16.py --check /train/datasets/bottle/images/all/coco_train2017_10799.jpg

keep_io_types=True で入出力契約は fp32 版と同一のまま
（input: uint8 RGBA / dets等: float32）。fp16 非対応 op（NMS 等）には
コンバータが自動で Cast を挿入する。
"""
import argparse

import numpy as np
import onnx


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/out/rtmdet_ins.onnx")
    ap.add_argument("--out", default="/out/rtmdet_ins_fp16.onnx")
    ap.add_argument("--check", metavar="IMG", default=None)
    args = ap.parse_args()

    from onnxconverter_common import float16

    model = onnx.load(args.model)
    # Cast ノードは to 属性が変換に追従せず型不整合になるため除外する
    # （境界にはコンバータが自動で Cast を挿入する）
    casts = [n.name for n in model.graph.node if n.op_type == "Cast"]
    model16 = float16.convert_float_to_float16(
        model, keep_io_types=True, node_block_list=casts)
    onnx.save(model16, args.out)

    import os
    print(f"[fp16] {args.out} "
          f"({os.path.getsize(args.out) // 1_000_000} MB, "
          f"fp32: {os.path.getsize(args.model) // 1_000_000} MB)")

    if args.check:
        import cv2
        import onnxruntime as ort

        size = [d.dim_value
                for d in model.graph.input[0].type.tensor_type.shape.dim][1]
        bgr = cv2.resize(cv2.imread(args.check), (size, size),
                         interpolation=cv2.INTER_LINEAR)
        rgba = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGBA)[None]

        providers = ["CPUExecutionProvider"]
        a = ort.InferenceSession(args.model, providers=providers).run(
            None, {"input": rgba})
        b = ort.InferenceSession(args.out, providers=providers).run(
            None, {"input": rgba})
        da, db = a[0][0], b[0][0]
        print(f"[check] dets fp32: {np.round(da[da[:, 4] > 0.3][:, 4], 3)}")
        print(f"[check] dets fp16: {np.round(db[db[:, 4] > 0.3][:, 4], 3)}")
        # fp16 は丸め誤差があるためスコアの近似一致を確認する
        ka = (da[:, 4] > 0.4).sum()
        kb = (db[:, 4] > 0.4).sum()
        if abs(int(ka) - int(kb)) > 1:
            raise SystemExit(f"[check] FAILED: det count fp32={ka} fp16={kb}")
        print("[check] OK")


if __name__ == "__main__":
    main()
