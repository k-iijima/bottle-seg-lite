#!/usr/bin/env bash
set -e

flutter config --enable-web >/dev/null 2>&1 || true

if [ ! -f assets/models/rtmdet_ins.onnx ]; then
  echo "============================================================"
  echo "ERROR: assets/models/rtmdet_ins.onnx not found."
  echo "Download it from Releases:"
  echo "  gh release download v0.2.0 -R k-iijima/bottle-seg-lite \\"
  echo "    -p rtmdet_ins.onnx -O app/assets/models/rtmdet_ins.onnx"
  echo "(or re-export from a trained ckpt: make rtmdet-onnx)"
  echo "============================================================"
  exit 1
fi

echo "[entrypoint] flutter pub get..."
flutter pub get

echo "[entrypoint] starting dev server on http://localhost:8080"
exec flutter run -d web-server --web-hostname 0.0.0.0 --web-port 8080
