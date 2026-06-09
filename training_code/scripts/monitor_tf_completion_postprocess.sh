#!/usr/bin/env bash
set -euo pipefail

cd /home/admin/Sais_ocr

INTERVAL="${INTERVAL:-300}"
TF_SESSION="${TF_SESSION:-classifier_queue_fixed}"
TF_RUN="${TF_RUN:-runs/classifier/tf_effv2s_224_fixed}"
CROP_DIR="${CROP_DIR:-work/classifier_crops_fixed_20260608}"
DETECTOR_WEIGHTS="${DETECTOR_WEIGHTS:-work/snapshots/yolo11l_selected_detector_best.pt}"
SUMMARY="${SUMMARY:-work/evals/pipeline_tf_effv2s_224_fixed_summary.json}"
EVAL_JSON="${EVAL_JSON:-work/evals/tf_effv2s_224_fixed_eval.json}"
REPORT="${REPORT:-work/evals/sync_h69_from_best_pipeline_report.json}"
LOG="${LOG:-work/logs/monitor_tf_completion_postprocess.log}"
CLS_BATCH="${CLS_BATCH:-96}"
WORKERS="${WORKERS:-8}"
DEVICE="${DEVICE:-cuda:0}"

mkdir -p "$(dirname "${LOG}")" work/evals

log() {
  echo "[$(date '+%F %T %Z')] $*" | tee -a "${LOG}"
}

latest_metrics() {
  if [[ -s "${TF_RUN}/metrics.csv" ]]; then
    python3 - "${TF_RUN}/metrics.csv" <<'PY' 2>/dev/null || true
import csv
import sys
from pathlib import Path

path = Path(sys.argv[1])
rows = list(csv.DictReader(path.open("r", encoding="utf-8", newline="")))
if not rows:
    raise SystemExit(0)
last = rows[-1]
best = max(rows, key=lambda row: float(row.get("val_top1", -1.0)))
print(
    "epochs={epochs} last_epoch={last_epoch} last_val_top1={last_top1} "
    "best_epoch={best_epoch} best_val_top1={best_top1}".format(
        epochs=len(rows),
        last_epoch=last.get("epoch"),
        last_top1=last.get("val_top1"),
        best_epoch=best.get("epoch"),
        best_top1=best.get("val_top1"),
    )
)
PY
  fi
}

log "watching ${TF_SESSION}; will postprocess only after it exits"

while tmux has-session -t "${TF_SESSION}" 2>/dev/null; do
  metric_line="$(latest_metrics || true)"
  if [[ -n "${metric_line}" ]]; then
    log "${TF_SESSION} still running; ${metric_line}"
  else
    log "${TF_SESSION} still running; metrics not ready"
  fi
  sleep "${INTERVAL}"
done

log "${TF_SESSION} is no longer running"

if [[ -s "${SUMMARY}" ]]; then
  log "TF pipeline summary already exists: ${SUMMARY}; running h69 sync gate"
  python3 code/sais_ocr_rebuild_20260608/scripts/sync_h69_from_best_pipeline.py \
    --summary work/evals/pipeline_effb0_128_fixed_summary.json \
    --summary "${SUMMARY}" \
    --report "${REPORT}" >> "${LOG}" 2>&1
  log "postprocess complete"
  exit 0
fi

if [[ ! -s "${TF_RUN}/best.pth" ]]; then
  log "missing TF best checkpoint: ${TF_RUN}/best.pth"
  exit 2
fi

if [[ ! -d "${CROP_DIR}/val" ]]; then
  log "missing classifier validation crop directory: ${CROP_DIR}/val"
  exit 2
fi

if [[ ! -s "${DETECTOR_WEIGHTS}" ]]; then
  log "missing detector weights: ${DETECTOR_WEIGHTS}"
  exit 2
fi

if [[ ! -s "${EVAL_JSON}" ]]; then
  log "running TF classifier crop eval"
  PYTHONUNBUFFERED=1 python3 code/sais_ocr_rebuild_20260608/scripts/eval_classifier.py \
    --checkpoint "${TF_RUN}/best.pth" \
    --data "${CROP_DIR}/val" \
    --out "${EVAL_JSON}" \
    --batch "${CLS_BATCH}" \
    --workers "${WORKERS}" \
    --device "${DEVICE}" >> "${LOG}" 2>&1
else
  log "classifier eval already exists: ${EVAL_JSON}"
fi

log "running TF full-pipeline grid"
PYTHONUNBUFFERED=1 python3 code/sais_ocr_rebuild_20260608/scripts/run_pipeline_grid.py \
  --det-weights "${DETECTOR_WEIGHTS}" \
  --cls-weights "${TF_RUN}/best.pth" \
  --tag "pipeline_tf_effv2s_224_fixed" \
  --confs 0.15,0.18,0.20,0.22,0.25,0.30 \
  --nms-ious 0.70 \
  --crop-paddings 0.05,0.10,0.15 \
  --imgsz 1536 \
  --max-det 600 \
  --device "${DEVICE}" \
  --det-batch 1 \
  --det-half \
  --cls-batch "${CLS_BATCH}" \
  --overwrite >> "${LOG}" 2>&1

if [[ ! -s "${SUMMARY}" ]]; then
  log "pipeline grid finished but summary is still missing: ${SUMMARY}"
  exit 3
fi

log "running h69 sync gate"
python3 code/sais_ocr_rebuild_20260608/scripts/sync_h69_from_best_pipeline.py \
  --summary work/evals/pipeline_effb0_128_fixed_summary.json \
  --summary "${SUMMARY}" \
  --report "${REPORT}" >> "${LOG}" 2>&1

log "postprocess complete"
