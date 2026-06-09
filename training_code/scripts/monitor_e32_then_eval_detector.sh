#!/usr/bin/env bash
set -euo pipefail

cd /home/admin/Sais_ocr

E32_SESSION="${E32_SESSION:-yolo11l_e32_safe}"
E32_DIR="${E32_DIR:-runs/detect/runs/detect/fixed_data_20260608/yolo11l_full_fixed_e32_safe_from_smoke}"
INTERVAL="${INTERVAL:-300}"
EVAL_LOG="${EVAL_LOG:-work/logs/yolo11l_e32_box_grid_eval.log}"
TAG="${TAG:-yolo11l_e32_safe_box}"

mkdir -p work/logs work/evals

echo "[$(date '+%F %T %Z')] waiting for ${E32_SESSION}"
while ! tmux has-session -t "${E32_SESSION}" 2>/dev/null; do
  echo "[$(date '+%F %T %Z')] e32 not started yet"
  sleep "${INTERVAL}"
done

while tmux has-session -t "${E32_SESSION}" 2>/dev/null; do
  echo "[$(date '+%F %T %Z')] e32 still running"
  sleep "${INTERVAL}"
done

RESULTS="${E32_DIR}/results.csv"
BEST="${E32_DIR}/weights/best.pt"
LAST="${E32_DIR}/weights/last.pt"

if [[ ! -s "${RESULTS}" || ! -s "${BEST}" || ! -s "${LAST}" ]]; then
  echo "[$(date '+%F %T %Z')] e32 missing required artifacts"
  echo "results=${RESULTS}"
  echo "best=${BEST}"
  echo "last=${LAST}"
  exit 1
fi

python3 - <<PY
import csv, math
path = "${RESULTS}"
with open(path, encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
if not rows:
    raise SystemExit(f"empty results: {path}")
for row in rows:
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
print("finite results:", path, "epochs:", len(rows))
PY

echo "[$(date '+%F %T %Z')] starting detector box grid, log=${EVAL_LOG}"
PYTHONUNBUFFERED=1 python3 code/sais_ocr_rebuild_20260608/scripts/run_detector_box_grid.py \
  --model "${BEST}" \
  --tag "${TAG}" \
  --confs 0.05,0.08,0.10,0.12,0.15,0.18,0.20,0.25,0.30 \
  --nms-ious 0.60,0.70 \
  --imgsz 1536 \
  --max-det 600 \
  --device 0 \
  --batch 1 \
  --half \
  --overwrite > "${EVAL_LOG}" 2>&1

echo "[$(date '+%F %T %Z')] detector box grid complete"
tail -20 "${EVAL_LOG}"
