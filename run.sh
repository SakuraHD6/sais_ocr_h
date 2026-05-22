#!/bin/bash
set -euo pipefail

echo "===== OCR Pipeline ====="

# If run with command args, execute them directly
if [ $# -gt 0 ]; then
    exec "$@"
fi

# Otherwise, run inference pipeline
echo "INPUT_DIR=${INPUT_DIR:-/saisdata/13/eval/images}"
echo "OUTPUT_FILE=${OUTPUT_FILE:-/saisresult/prediction.json}"

# GPU setup
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ] && [ -n "${NVIDIA_VISIBLE_DEVICES:-}" ] \
  && [ "${NVIDIA_VISIBLE_DEVICES}" != "all" ] && [ "${NVIDIA_VISIBLE_DEVICES}" != "void" ]; then
  export CUDA_VISIBLE_DEVICES="${NVIDIA_VISIBLE_DEVICES}"
fi

# Dynamically link host's libcuda.so at runtime (forward-compat for older drivers)
mkdir -p /usr/local/cuda/lib64 || true
for lib in libcuda.so libnvidia-ml.so; do
  if [ ! -e "/usr/local/cuda/lib64/${lib}" ]; then
    target="$(ldconfig -p 2>/dev/null | awk -v name="${lib}.1" '$1 == name && $NF !~ "/usr/local/cuda/compat/" {print $NF; exit}' || true)"
    if [ -n "${target}" ]; then
      ln -sf "${target}" "/usr/local/cuda/lib64/${lib}" || true
    fi
  fi
done
ldconfig || true

python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Devices: {torch.cuda.device_count()}')"

python3 /app/src/run_inference.py

echo "Done!"
