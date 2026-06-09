#!/usr/bin/env bash
set -euo pipefail

cd /home/admin/Sais_ocr

DETECTOR_SESSION="${DETECTOR_SESSION:-yolo11l_e32_safe}"
DETECTOR_EVAL_SESSION="${DETECTOR_EVAL_SESSION:-yolo11l_e32_then_eval}"
DETECTOR_EVAL_SUMMARY="${DETECTOR_EVAL_SUMMARY:-work/evals/yolo11l_e32_safe_box_summary.json}"
INTERVAL="${INTERVAL:-300}"
CROP_DIR="${CROP_DIR:-work/classifier_crops_fixed_20260608}"
CROP_LOG="${CROP_LOG:-work/logs/prepare_classifier_crops_fixed_20260608.log}"
CROP_SESSION="${CROP_SESSION:-prepare_classifier_crops_fixed}"
CLASSIFIER_SESSION="${CLASSIFIER_SESSION:-classifier_queue_fixed}"
CLASSIFIER_LOG="${CLASSIFIER_LOG:-work/logs/classifier_queue_fixed.log}"
DETECTOR_WEIGHTS="${DETECTOR_WEIGHTS:-work/snapshots/yolo11l_selected_detector_best.pt}"

mkdir -p work/logs

echo "[$(date '+%F %T %Z')] waiting for detector session ${DETECTOR_SESSION}"
while tmux has-session -t "${DETECTOR_SESSION}" 2>/dev/null; do
  echo "[$(date '+%F %T %Z')] detector still running"
  sleep "${INTERVAL}"
done

if [[ -n "${DETECTOR_EVAL_SESSION}" ]]; then
  echo "[$(date '+%F %T %Z')] waiting for detector eval session ${DETECTOR_EVAL_SESSION}"
  while tmux has-session -t "${DETECTOR_EVAL_SESSION}" 2>/dev/null; do
    echo "[$(date '+%F %T %Z')] detector eval still running"
    sleep "${INTERVAL}"
  done
fi

if [[ ! -s "${DETECTOR_WEIGHTS}" ]]; then
  echo "[$(date '+%F %T %Z')] selected detector checkpoint missing; not starting classifier prep: ${DETECTOR_WEIGHTS}"
  exit 1
fi

if [[ -n "${DETECTOR_EVAL_SUMMARY}" && ! -s "${DETECTOR_EVAL_SUMMARY}" ]]; then
  echo "[$(date '+%F %T %Z')] detector eval summary missing; not starting classifier prep: ${DETECTOR_EVAL_SUMMARY}"
  exit 1
fi

while [[ ! -s "${CROP_DIR}/stats.json" && "${CROP_SESSION}" != "" ]] && tmux has-session -t "${CROP_SESSION}" 2>/dev/null; do
  echo "[$(date '+%F %T %Z')] existing classifier crop session still running: ${CROP_SESSION}"
  sleep "${INTERVAL}"
done

if [[ ! -s "${CROP_DIR}/stats.json" ]]; then
  echo "[$(date '+%F %T %Z')] preparing classifier crops -> ${CROP_DIR}"
  PYTHONUNBUFFERED=1 python3 code/sais_ocr_rebuild_20260608/scripts/prepare_classifier_crops.py \
    --out "${CROP_DIR}" > "${CROP_LOG}" 2>&1
else
  echo "[$(date '+%F %T %Z')] classifier crops already exist: ${CROP_DIR}/stats.json"
fi

python3 - <<PY
import json
from pathlib import Path
stats = Path("${CROP_DIR}") / "stats.json"
mapping = Path("${CROP_DIR}") / "class_mapping.json"
if not stats.exists() or not mapping.exists():
    raise SystemExit("missing classifier crop stats or mapping")
data = json.loads(stats.read_text(encoding="utf-8"))
print("classifier crop stats:", json.dumps(data, ensure_ascii=False, indent=2))
if data.get("labels", 0) < 4000:
    raise SystemExit(f"unexpectedly low label count: {data.get('labels')}")
PY

if tmux has-session -t "${CLASSIFIER_SESSION}" 2>/dev/null; then
  echo "[$(date '+%F %T %Z')] classifier queue already running: ${CLASSIFIER_SESSION}"
  exit 0
fi

echo "[$(date '+%F %T %Z')] starting classifier queue session ${CLASSIFIER_SESSION}"
tmux new -d -s "${CLASSIFIER_SESSION}" \
  "cd /home/admin/Sais_ocr && PYTHONUNBUFFERED=1 CROP_DIR='${CROP_DIR}' DETECTOR_SESSION='' DETECTOR_EVAL_SESSION='' DETECTOR_WEIGHTS='${DETECTOR_WEIGHTS}' bash code/sais_ocr_rebuild_20260608/scripts/train_classifier_queue.sh > ${CLASSIFIER_LOG} 2>&1"

echo "[$(date '+%F %T %Z')] classifier queue started, log=${CLASSIFIER_LOG}"
