#!/usr/bin/env bash
set -euo pipefail

cd /home/admin/Sais_ocr

INTERVAL="${INTERVAL:-300}"
LOG="${LOG:-work/logs/rebuild_status_monitor_300s.log}"
CROP_DIR="${CROP_DIR:-work/classifier_crops_fixed_20260608}"
EPOCH5_SUMMARY="${EPOCH5_SUMMARY:-work/evals/yolo11l_e32_epoch5_box_summary.json}"
E32_SUMMARY="${E32_SUMMARY:-work/evals/yolo11l_e32_safe_box_summary.json}"

mkdir -p "$(dirname "${LOG}")"

while true; do
  {
    echo "===== $(date '+%F %T %Z') ====="
    python3 code/sais_ocr_rebuild_20260608/scripts/show_rebuild_status.py --json || true
    echo
    echo "disk:"
    df -h /home/admin/Sais_ocr || true
    echo
    echo "classifier_crops:"
    du -sh "${CROP_DIR}" 2>/dev/null || true
    find "${CROP_DIR}" -type f 2>/dev/null | wc -l | awk '{print "files=" $1}' || true
    echo
    echo "epoch5_summary:"
    if [[ -s "${EPOCH5_SUMMARY}" ]]; then
      python3 - <<PY
import json
from pathlib import Path
path = Path("${EPOCH5_SUMMARY}")
data = json.loads(path.read_text(encoding="utf-8"))
print(json.dumps(data[:5] if isinstance(data, list) else data, ensure_ascii=False, indent=2))
PY
    else
      echo "missing"
    fi
    echo
    echo "e32_summary:"
    if [[ -s "${E32_SUMMARY}" ]]; then
      python3 - <<PY
import json
from pathlib import Path
path = Path("${E32_SUMMARY}")
data = json.loads(path.read_text(encoding="utf-8"))
print(json.dumps(data[:5] if isinstance(data, list) else data, ensure_ascii=False, indent=2))
PY
    else
      echo "missing"
    fi
    echo
  } >> "${LOG}" 2>&1
  sleep "${INTERVAL}"
done
