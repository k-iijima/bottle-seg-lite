"""mmdeploy 出力の RTMDet-Ins ONNX に前処理を埋め込む（export_rtmdet.sh から呼ばれる）。

入力契約を
  float32 [1,3,S,S]  BGR・mean/std 正規化済み（アプリ側で per-pixel 変換が必要）
から
  uint8   [1,S,S,4]  RGBA そのまま（canvas getImageData / カメラフレーム直渡し）
に変える。RGBA→BGR の並べ替え・float 化・正規化はグラフ先頭の
Cast→Transpose→Gather→Sub→Div で行う（alpha は Gather で捨てる）。

使い方: python embed_preprocess.py IN.onnx OUT.onnx [--check DEMO_IMG]
--check を付けると、元モデル（float 前処理を numpy で再現）と埋め込み後モデル
（uint8 直渡し）の dets を比較して一致を確認する。
"""
import argparse

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

# detector.dart / RTMDet data_preprocessor と同値（BGR 順）
MEAN_BGR = [103.53, 116.28, 123.675]
STD_BGR = [57.375, 57.12, 58.395]


def embed(model: onnx.ModelProto) -> onnx.ModelProto:
    g = model.graph
    old_in = next(i for i in g.input if i.name == "input")
    dims = [d.dim_value for d in old_in.type.tensor_type.shape.dim]
    assert dims[1] == 3, f"expected NCHW float input, got {dims}"
    s = dims[2]

    # 既存ノードの 'input' 参照を正規化済みテンソル名に付け替え
    for n in g.node:
        for i, name in enumerate(n.input):
            if name == "input":
                n.input[i] = "pp_norm"

    g.input.remove(old_in)
    g.input.insert(
        0, helper.make_tensor_value_info("input", TensorProto.UINT8, [1, s, s, 4])
    )

    g.initializer.extend([
        numpy_helper.from_array(
            np.array(MEAN_BGR, np.float32).reshape(1, 3, 1, 1), "pp_mean"),
        numpy_helper.from_array(
            np.array(STD_BGR, np.float32).reshape(1, 3, 1, 1), "pp_std"),
        # NCHW 化した RGBA のチャネル [R,G,B,A] から [B,G,R] を選ぶ
        numpy_helper.from_array(np.array([2, 1, 0], np.int64), "pp_bgr_idx"),
    ])

    pp_nodes = [
        helper.make_node("Cast", ["input"], ["pp_f32"], name="pp_cast",
                         to=TensorProto.FLOAT),
        helper.make_node("Transpose", ["pp_f32"], ["pp_nchw"], name="pp_transpose",
                         perm=[0, 3, 1, 2]),
        helper.make_node("Gather", ["pp_nchw", "pp_bgr_idx"], ["pp_bgr"],
                         name="pp_gather", axis=1),
        helper.make_node("Sub", ["pp_bgr", "pp_mean"], ["pp_sub"], name="pp_sub"),
        helper.make_node("Div", ["pp_sub", "pp_std"], ["pp_norm"], name="pp_div"),
    ]
    for i, n in enumerate(pp_nodes):
        g.node.insert(i, n)

    # mmdeploy は動的な出力次元を dim_value=0 で宣言する（dets [1,0,5] 等）。
    # サイズ 0 の静的テンソルと解釈され、NNAPI 等の EP パーティショナが
    # セッション作成中に abort する原因になるため、シンボリック次元に直す。
    for out in g.output:
        for i, d in enumerate(out.type.tensor_type.shape.dim):
            if d.dim_value == 0 and not d.dim_param:
                d.Clear()
                d.dim_param = f"{out.name}_dyn{i}"
    return model


def check(orig_path: str, new_path: str, img_path: str, size: int) -> None:
    import cv2
    import onnxruntime as ort

    bgr = cv2.resize(cv2.imread(img_path), (size, size),
                     interpolation=cv2.INTER_LINEAR)
    rgba = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGBA)[None]  # [1,S,S,4] uint8

    x = (bgr.astype(np.float32) - np.array(MEAN_BGR, np.float32)) \
        / np.array(STD_BGR, np.float32)
    x = x.transpose(2, 0, 1)[None]  # [1,3,S,S] float32

    providers = ["CPUExecutionProvider"]
    a = ort.InferenceSession(orig_path, providers=providers).run(None, {"input": x})
    b = ort.InferenceSession(new_path, providers=providers).run(
        None, {"input": rgba})

    da, db = a[0][0], b[0][0]
    print(f"[check] dets fp32-pre : {np.round(da[da[:, 4] > 0.3][:, 4], 3)}")
    print(f"[check] dets embedded : {np.round(db[db[:, 4] > 0.3][:, 4], 3)}")
    if not np.allclose(da, db, atol=1e-3):
        raise SystemExit("[check] FAILED: outputs differ beyond tolerance")
    print("[check] OK: outputs match (atol=1e-3)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--check", metavar="IMG", default=None)
    args = ap.parse_args()

    model = onnx.load(args.src)
    size = [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim][2]

    import tempfile
    orig_copy = None
    if args.check:
        # 埋め込みで src を上書きする場合に備え、比較用に元モデルを退避
        orig_copy = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False).name
        onnx.save(model, orig_copy)

    embedded = embed(onnx.load(args.src))
    try:
        onnx.checker.check_model(embedded)
    except Exception as e:  # mmdeploy カスタム op ドメインは checker 非対応
        print(f"[embed] checker warning (ignored): {e}")
    onnx.save(embedded, args.dst)
    print(f"[embed] input contract -> uint8 [1,{size},{size},4] RGBA (NHWC)")

    if args.check:
        check(orig_copy, args.dst, args.check, size)


if __name__ == "__main__":
    main()
