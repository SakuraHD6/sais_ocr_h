#!/usr/bin/env bash
set -euo pipefail

cd /home/admin/Sais_ocr

INTERVAL="${INTERVAL:-300}"
TF_SESSION="${TF_SESSION:-classifier_queue_fixed}"
SUMMARY="${SUMMARY:-work/evals/pipeline_tf_effv2s_224_fixed_summary.json}"
LOG="${LOG:-work/logs/monitor_tf_pipeline_then_sync_h69.log}"
REPORT="${REPORT:-work/evals/sync_h69_from_best_pipeline_report.json}"

mkdir -p "$(dirname "${LOG}")" "$(dirname "${REPORT}")"

echo "[$(date '+%F %T %Z')] waiting for TF pipeline summary: ${SUMMARY}" >> "${LOG}"

while [[ ! -s "${SUMMARY}" ]]; do
  if tmux has-session -t "${TF_SESSION}" 2>/dev/null; then
    echo "[$(date '+%F %T %Z')] ${TF_SESSION} still running; summary not ready" >> "${LOG}"
  else
    echo "[$(date '+%F %T %Z')] ${TF_SESSION} not running; summary still missing" >> "${LOG}"
  fi
  sleep "${INTERVAL}"
done

echo "[$(date '+%F %T %Z')] TF pipeline summary ready; syncing h69 only if improved" >> "${LOG}"
python3 code/sais_ocr_rebuild_20260608/scripts/sync_h69_from_best_pipeline.py \
  --summary work/evals/pipeline_effb0_128_fixed_summary.json \
  --summary "${SUMMARY}" \
  --report "${REPORT}" >> "${LOG}" 2>&1

echo "[$(date '+%F %T %Z')] monitor complete" >> "${LOG}"
