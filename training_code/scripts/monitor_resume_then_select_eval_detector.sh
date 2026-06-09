#!/usr/bin/env bash
set -euo pipefail

cd /home/admin/Sais_ocr

DETECTOR_SESSION="${DETECTOR_SESSION:-yolo11l_e32_resume_b1}"
INTERVAL="${INTERVAL:-300}"
SELECTED_WEIGHTS="${SELECTED_WEIGHTS:-work/snapshots/yolo11l_selected_detector_best.pt}"
SELECT_REPORT="${SELECT_REPORT:-work/evals/yolo11l_selected_detector_best.json}"
EVAL_LOG="${EVAL_LOG:-work/logs/yolo11l_selected_detector_box_grid_eval.log}"
TAG="${TAG:-yolo11l_selected_detector_box}"

mkdir -p work/logs work/evals work/snapshots

echo "[$(date '+%F %T %Z')] waiting for detector session ${DETECTOR_SESSION}"
while ! tmux has-session -t "${DETECTOR_SESSION}" 2>/dev/null; do
  echo "[$(date '+%F %T %Z')] detector session not started yet"
  sleep "${INTERVAL}"
done

while tmux has-session -t "${DETECTOR_SESSION}" 2>/dev/null; do
  echo "[$(date '+%F %T %Z')] detector still running"
  sleep "${INTERVAL}"
done

echo "[$(date '+%F %T %Z')] selecting detector checkpoint"
PYTHONUNBUFFERED=1 python3 code/sais_ocr_rebuild_20260608/scripts/select_detector_by_box_grid.py \
  --out "${SELECTED_WEIGHTS}" \
  --report "${SELECT_REPORT}" \
  --summary "work/evals/${TAG}_summary.json" \
  --tag "${TAG}" \
  --confs 0.08,0.10,0.12,0.15,0.18,0.20,0.25,0.30 \
  --nms-ious 0.60,0.70 \
  --imgsz 1536 \
  --max-det 500 \
  --device 0 \
  --batch 1 \
  --half \
  --overwrite > "${EVAL_LOG}" 2>&1

if [[ ! -s "${SELECTED_WEIGHTS}" || ! -s "${SELECT_REPORT}" ]]; then
  echo "[$(date '+%F %T %Z')] selected detector artifacts missing"
  exit 1
fi

if [[ ! -s "work/evals/${TAG}_summary.json" ]]; then
  echo "[$(date '+%F %T %Z')] detector box grid summary missing: work/evals/${TAG}_summary.json"
  exit 1
fi

echo "[$(date '+%F %T %Z')] selected detector multi-candidate box grid complete"
tail -20 "${EVAL_LOG}"
