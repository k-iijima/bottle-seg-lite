#!/usr/bin/env bash
set -e

flutter config --enable-web >/dev/null 2>&1 || true

if [ ! -f assets/models/seg.onnx ]; then
  echo "============================================================"
  echo "ERROR: assets/models/seg.onnx not found."
  echo "Export the model first:  make model   (or: docker compose --profile tools run --rm model)"
  echo "============================================================"
  exit 1
fi

echo "[entrypoint] flutter pub get..."
flutter pub get

echo "[entrypoint] starting dev server on http://localhost:8080"
exec flutter run -d web-server --web-hostname 0.0.0.0 --web-port 8080
