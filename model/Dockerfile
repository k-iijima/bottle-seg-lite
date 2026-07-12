# Exports the ONNX segmentation model used by the Flutter web app.
# CPU-only torch/torchvision keeps the image small enough for a one-off export.
FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# CPU wheels from the official PyTorch index (torch/torchvision), plus the
# `onnx` package which torch.onnx.export needs at export time.
RUN pip install \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.3.1 torchvision==0.18.1 \
 && pip install onnx==1.16.1

WORKDIR /work
COPY export_model.py .

# OUT_PATH is mounted to the app's assets dir via docker-compose.
ENV OUT_PATH=/out/seg.onnx \
    INPUT_SIZE=256

CMD ["python", "export_model.py"]
