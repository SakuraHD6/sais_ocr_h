#!/usr/bin/env bash
set -euo pipefail

cd /home/admin/Sais_ocr

SMOKE_SESSION="${SMOKE_SESSION:-yolo11l_e1_smoke}"
SMOKE_DIR="${SMOKE_DIR:-runs/detect/runs/detect/fixed_data_20260608/yolo11l_full_fixed_e1_smoke_tmux}"
E32_SESSION="${E32_SESSION:-yolo11l_e32_safe}"
E32_LOG="${E32_LOG:-work/logs/yolo11l_full_fixed_e32_safe.log}"
E32_NAME="${E32_NAME:-yolo11l_full_fixed_e32_safe_from_smoke}"
INTERVAL="${INTERVAL:-300}"

mkdir -p work/logs

echo "[$(date '+%F %T %Z')] waiting for ${SMOKE_SESSION}"
while tmux has-session -t "${SMOKE_SESSION}" 2>/dev/null; do
  echo "[$(date '+%F %T %Z')] smoke still running"
  sleep "${INTERVAL}"
done

RESULTS="${SMOKE_DIR}/results.csv"
BEST="${SMOKE_DIR}/weights/best.pt"
LAST="${SMOKE_DIR}/weights/last.pt"

if [[ ! -s "${RESULTS}" || ! -s "${BEST}" || ! -s "${LAST}" ]]; then
  echo "[$(date '+%F %T %Z')] smoke missing required artifacts"
  echo "results=${RESULTS}"
  echo "best=${BEST}"
  echo "last=${LAST}"
  exit 1
fi

python3 - <<PY
import csv, math
path = "${RESULTS}"
with open(path, encoding="utf-8") as f:
    for row in csv.DictReader(f):
        for key, value in row.items():
            value = (value or "").strip()
            if not value:
                continue
            try:
                number = float(value)
            except ValueError:
                continue
            if not math.isfinite(number):
                raise SystemExit(f"non-finite {key}={value} in {path}")
print("finite results:", path)
PY

if tmux has-session -t "${E32_SESSION}" 2>/dev/null; then
  echo "[$(date '+%F %T %Z')] ${E32_SESSION} already running"
  exit 0
fi

echo "[$(date '+%F %T %Z')] starting ${E32_SESSION}"
tmux new -d -s "${E32_SESSION}" \
  "cd /home/admin/Sais_ocr && PYTHONUNBUFFERED=1 YOLO_CONFIG_DIR=/tmp/Ultralytics python3 code/sais_ocr_rebuild_20260608/scripts/train_yolo_safe.py \
    --data work/yolo_data_fixed_20260608/data.yaml \
    --model ${BEST} \
    --project runs/detect/fixed_data_20260608 \
    --name ${E32_NAME} \
    --imgsz 1536 --epochs 32 --batch 2 --device 0 --workers 4 \
    --optimizer AdamW --lr0 0.0001 --lrf 0.05 \
    --weight-decay 0.0005 --patience 12 --save-period 5 \
    --exist-ok --check-finite > ${E32_LOG} 2>&1"

echo "[$(date '+%F %T %Z')] started ${E32_SESSION}, log=${E32_LOG}"

