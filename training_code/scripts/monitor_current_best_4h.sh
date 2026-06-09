#!/usr/bin/env bash
set -euo pipefail

cd /home/admin/Sais_ocr

INTERVAL="${INTERVAL:-300}"
DURATION_SECONDS="${DURATION_SECONDS:-14400}"
LOG="${LOG:-work/logs/current_best_snapshot_4h_20260609.log}"
SCRIPT="code/sais_ocr_rebuild_20260608/scripts/snapshot_current_best.py"

mkdir -p "$(dirname "${LOG}")"

start_ts="$(date +%s)"
end_ts="$((start_ts + DURATION_SECONDS))"

echo "===== current best snapshot monitor started $(date '+%F %T %Z') =====" >> "${LOG}"
echo "interval=${INTERVAL} duration_seconds=${DURATION_SECONDS}" >> "${LOG}"

while true; do
  now_ts="$(date +%s)"
  if (( now_ts > end_ts )); then
    break
  fi

  {
    echo
    echo "===== $(date '+%F %T %Z') ====="
    python3 "${SCRIPT}" --append-status
  } >> "${LOG}" 2>&1 || true

  now_ts="$(date +%s)"
  if (( now_ts >= end_ts )); then
    break
  fi
  sleep "${INTERVAL}"
done

{
  echo
  echo "===== final refresh $(date '+%F %T %Z') ====="
  python3 "${SCRIPT}" --append-status
  echo "===== current best snapshot monitor finished $(date '+%F %T %Z') ====="
} >> "${LOG}" 2>&1 || true
