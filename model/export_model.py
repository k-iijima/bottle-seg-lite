"""Export a lightweight segmentation model to ONNX for the Flutter web demo.

We use torchvision's LR-ASPP MobileNetV3-Large (Pascal VOC, 21 classes).
It is small (~12MB) and mobile-oriented, which keeps the browser download
and onnxruntime-web inference reasonable.

The important part for the Flutter side is the *I/O contract*, which we fix
explicitly so the web (JavaScript) ONNX Runtime API does not need metadata:

    input  : float32  [1, 3, H, W]   NCHW, ImageNet-normalized RGB
    output : float32  [1, 21, H, W]   per-class logits (argmax over dim=1)

When you replace this with your own model later, keep the same names
('input' / 'output') OR update INPUT_NAME / OUTPUT_NAME in lib/segmenter.dart.
"""

import os

import torch
import torch.nn as nn
from torchvision.models.segmentation import (
    LRASPP_MobileNet_V3_Large_Weights,
    lraspp_mobilenet_v3_large,
)

INPUT_SIZE = int(os.environ.get("INPUT_SIZE", "256"))
OUT_PATH = os.environ.get("OUT_PATH", "/out/seg.onnx")
OPSET = 17


class SegWrapper(nn.Module):
    """Unwrap the torchvision OrderedDict output to a single logits tensor."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)["out"]


def main() -> None:
    weights = LRASPP_MobileNet_V3_Large_Weights.DEFAULT
    model = lraspp_mobilenet_v3_large(weights=weights)
    model.eval()

    wrapped = SegWrapper(model).eval()
    dummy = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    torch.onnx.export(
        wrapped,
        dummy,
        OUT_PATH,
        export_params=True,
        opset_version=OPSET,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        # Allow swapping input resolution without re-exporting.
        dynamic_axes={
            "input": {0: "batch", 2: "height", 3: "width"},
            "output": {0: "batch", 2: "height", 3: "width"},
        },
    )

    size_mb = os.path.getsize(OUT_PATH) / (1024 * 1024)
    print(f"[export] wrote {OUT_PATH} ({size_mb:.1f} MB)")
    print(f"[export] input='input' [1,3,{INPUT_SIZE},{INPUT_SIZE}]  output='output' [1,21,H,W]")
    print(f"[export] classes (VOC): {weights.meta['categories']}")


if __name__ == "__main__":
    main()
