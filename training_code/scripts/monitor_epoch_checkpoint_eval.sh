#!/usr/bin/env bash
set -euo pipefail

cd /home/admin/Sais_ocr

RUN_DIR="${RUN_DIR:-runs/detect/runs/detect/fixed_data_20260608/yolo11l_full_fixed_e32_safe_from_smoke}"
MIN_RESULT_ROWS="${MIN_RESULT_ROWS:-5}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-best.pt}"
TAG="${TAG:-yolo11l_e32_epoch5_box}"
INTERVAL="${INTERVAL:-300}"
EVAL_LOG="${EVAL_LOG:-work/logs/yolo11l_e32_epoch5_box_grid_eval.log}"
EVAL_DEVICE="${EVAL_DEVICE:-cpu}"
EVAL_BATCH="${EVAL_BATCH:-1}"
SNAPSHOT="${SNAPSHOT:-work/snapshots/yolo11l_e32_after${MIN_RESULT_ROWS}_best.pt}"

mkdir -p work/logs work/evals work/snapshots

CHECKPOINT="${RUN_DIR}/weights/${CHECKPOINT_NAME}"
RESULTS="${RUN_DIR}/results.csv"

echo "[$(date '+%F %T %Z')] waiting for ${RESULTS} to have at least ${MIN_RESULT_ROWS} rows and ${CHECKPOINT}"
while true; do
  if [[ -s "${RESULTS}" && -s "${CHECKPOINT}" ]]; then
    ROWS="$(python3 - <<PY
import csv
from pathlib import Path
path = Path("${RESULTS}")
with path.open(encoding="utf-8") as f:
    print(len(list(csv.DictReader(f))))
PY
)"
    if [[ "${ROWS}" -ge "${MIN_RESULT_ROWS}" ]]; then
      break
    fi
    echo "[$(date '+%F %T %Z')] results rows=${ROWS}, need ${MIN_RESULT_ROWS}"
  else
    echo "[$(date '+%F %T %Z')] waiting artifacts: results=${RESULTS} checkpoint=${CHECKPOINT}"
  fi
  sleep "${INTERVAL}"
done

python3 - <<PY
import csv, math
path = "${RESULTS}"
with open(path, encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
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
print("finite results:", path, "rows:", len(rows))
PY

cp -f "${CHECKPOINT}" "${SNAPSHOT}"

echo "[$(date '+%F %T %Z')] starting checkpoint detector box grid, log=${EVAL_LOG}"
PYTHONUNBUFFERED=1 python3 code/sais_ocr_rebuild_20260608/scripts/run_detector_box_grid.py \
  --model "${SNAPSHOT}" \
  --tag "${TAG}" \
  --confs 0.05,0.08,0.10,0.12,0.15,0.18,0.20,0.25,0.30 \
  --nms-ious 0.60,0.70 \
  --imgsz 1536 \
  --max-det 600 \
  --device "${EVAL_DEVICE}" \
  --batch "${EVAL_BATCH}" \
  --overwrite > "${EVAL_LOG}" 2>&1

echo "[$(date '+%F %T %Z')] checkpoint detector box grid complete"
tail -20 "${EVAL_LOG}"
