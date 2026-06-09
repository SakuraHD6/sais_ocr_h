#!/usr/bin/env bash
set -euo pipefail

cd /home/admin/Sais_ocr

CROP_DIR="${CROP_DIR:-work/classifier_crops_fixed_20260608}"
DETECTOR_SESSION="${DETECTOR_SESSION:-yolo11l_e32_safe}"
DETECTOR_EVAL_SESSION="${DETECTOR_EVAL_SESSION:-yolo11l_e32_then_eval}"
DETECTOR_WEIGHTS="${DETECTOR_WEIGHTS:-work/snapshots/yolo11l_selected_detector_best.pt}"
INTERVAL="${INTERVAL:-300}"
CANDIDATE_OUT="${CANDIDATE_OUT:-local/sais_ocr_rebuild_20260608_candidate}"
CANDIDATE_REPORT="${CANDIDATE_REPORT:-work/evals/pipeline_best_candidate_report.json}"
MIN_CANDIDATE_E2E_F1="${MIN_CANDIDATE_E2E_F1:-0.60}"
mkdir -p work/logs runs/classifier
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

echo "[$(date '+%F %T %Z')] waiting for classifier crops: ${CROP_DIR}/stats.json"
while [[ ! -s "${CROP_DIR}/stats.json" || ! -s "${CROP_DIR}/class_mapping.json" ]]; do
  echo "[$(date '+%F %T %Z')] classifier crops not ready"
  sleep "${INTERVAL}"
done

if [[ -n "${DETECTOR_SESSION}" ]]; then
  echo "[$(date '+%F %T %Z')] waiting for detector session before classifier GPU training: ${DETECTOR_SESSION}"
  while tmux has-session -t "${DETECTOR_SESSION}" 2>/dev/null; do
    echo "[$(date '+%F %T %Z')] detector still running; classifier queue waits"
    sleep "${INTERVAL}"
  done
fi

if [[ -n "${DETECTOR_EVAL_SESSION}" ]]; then
  echo "[$(date '+%F %T %Z')] waiting for detector eval session before classifier GPU training: ${DETECTOR_EVAL_SESSION}"
  while tmux has-session -t "${DETECTOR_EVAL_SESSION}" 2>/dev/null; do
    echo "[$(date '+%F %T %Z')] detector eval still running; classifier queue waits"
    sleep "${INTERVAL}"
  done
fi

python3 - <<PY
import json
from pathlib import Path
stats = Path("${CROP_DIR}") / "stats.json"
data = json.loads(stats.read_text(encoding="utf-8"))
print("classifier crop stats:", json.dumps(data, ensure_ascii=False, indent=2), flush=True)
if data.get("labels", 0) < 4000:
    raise SystemExit(f"unexpectedly low label count: {data.get('labels')}")
for split in ("train", "val"):
    written = data.get("splits", {}).get(split, {}).get("written", 0)
    if written <= 0:
        raise SystemExit(f"no {split} classifier crops written")
PY

run_classifier() {
  local name="$1"
  local model="$2"
  local img_size="$3"
  local epochs="$4"
  local batch="$5"
  local workers="$6"
  local lr="$7"
  local patience="$8"
  local out="runs/classifier/${name}"
  local log="work/logs/${name}.log"
  local eval_json="work/evals/${name}_eval.json"
  local min_existing_epochs=5

  if [[ -s "${out}/best.pth" ]]; then
    if python3 - "${out}" "${model}" "${img_size}" "${min_existing_epochs}" <<'PY'
import csv
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
expected_model = sys.argv[2]
expected_img_size = int(sys.argv[3])
min_epochs = int(sys.argv[4])

config_path = out / "config.json"
metrics_path = out / "metrics.csv"
best_path = out / "best.pth"
if not config_path.exists() or not metrics_path.exists() or not best_path.exists():
    raise SystemExit(1)

config = json.loads(config_path.read_text(encoding="utf-8"))
if config.get("model") != expected_model or int(config.get("img_size", 0)) != expected_img_size:
    raise SystemExit(1)
if int(config.get("classes", 0)) < 4000:
    raise SystemExit(1)

with metrics_path.open("r", encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f))
if len(rows) < min_epochs:
    raise SystemExit(1)
if not rows:
    raise SystemExit(1)

try:
    best_top1 = max(float(row.get("val_top1", "nan")) for row in rows)
except ValueError:
    raise SystemExit(1)
if best_top1 != best_top1:
    raise SystemExit(1)
PY
    then
      echo "[$(date '+%F %T %Z')] ${name} already has a usable checkpoint, skipping train"
    else
      local stamp
      stamp="$(date '+%Y%m%d_%H%M%S')"
      local backup="work/snapshots/interrupted_${name}_${stamp}"
      echo "[$(date '+%F %T %Z')] ${name} checkpoint is incomplete or stale; moving to ${backup}"
      mkdir -p "$(dirname "${backup}")"
      mv "${out}" "${backup}"
    fi
  fi

  if [[ -s "${out}/best.pth" ]]; then
    :
  else
    echo "[$(date '+%F %T %Z')] training classifier ${name}, log=${log}"
    PYTHONUNBUFFERED=1 python3 code/sais_ocr_rebuild_20260608/scripts/train_classifier.py \
      --data "${CROP_DIR}" \
      --out "${out}" \
      --model "${model}" \
      --img-size "${img_size}" \
      --epochs "${epochs}" \
      --batch "${batch}" \
      --workers "${workers}" \
      --lr "${lr}" \
      --device cuda:0 \
      --pretrained \
      --pretrained-fallback \
      --amp \
      --weighted-sampler \
      --patience "${patience}" \
      --save-every 5 > "${log}" 2>&1
  fi

  if [[ -s "${out}/best.pth" ]]; then
    echo "[$(date '+%F %T %Z')] evaluating classifier ${name}"
    PYTHONUNBUFFERED=1 python3 code/sais_ocr_rebuild_20260608/scripts/eval_classifier.py \
      --checkpoint "${out}/best.pth" \
      --data "${CROP_DIR}/val" \
      --out "${eval_json}" \
      --batch "${batch}" \
      --workers "${workers}" \
      --device cuda:0

    if [[ -s "${DETECTOR_WEIGHTS}" ]]; then
      echo "[$(date '+%F %T %Z')] running pipeline grid for ${name}"
      PYTHONUNBUFFERED=1 python3 code/sais_ocr_rebuild_20260608/scripts/run_pipeline_grid.py \
        --det-weights "${DETECTOR_WEIGHTS}" \
        --cls-weights "${out}/best.pth" \
        --tag "pipeline_${name}" \
        --confs 0.15,0.18,0.20,0.22,0.25,0.30 \
        --nms-ious 0.70 \
        --crop-paddings 0.05,0.10,0.15 \
        --imgsz 1536 \
        --max-det 600 \
        --device cuda:0 \
        --det-batch 1 \
        --det-half \
        --cls-batch "${batch}" \
        --overwrite
    else
      echo "[$(date '+%F %T %Z')] detector weights missing, skip pipeline grid: ${DETECTOR_WEIGHTS}"
    fi
  else
    echo "[$(date '+%F %T %Z')] missing best checkpoint after ${name}"
    return 1
  fi
}

run_classifier "effb0_128_fixed" "efficientnet_b0" "128" "60" "256" "8" "0.0003" "12"
run_classifier "tf_effv2s_224_fixed" "tf_efficientnetv2_s" "224" "80" "96" "8" "0.0002" "16"

echo "[$(date '+%F %T %Z')] selecting best pipeline summary and applying candidate gate"
PYTHONUNBUFFERED=1 python3 code/sais_ocr_rebuild_20260608/scripts/select_pipeline_candidate.py \
  --summary work/evals/pipeline_effb0_128_fixed_summary.json \
  --summary work/evals/pipeline_tf_effv2s_224_fixed_summary.json \
  --report "${CANDIDATE_REPORT}" \
  --out "${CANDIDATE_OUT}" \
  --min-e2e-f1 "${MIN_CANDIDATE_E2E_F1}" \
  --overwrite

echo "[$(date '+%F %T %Z')] classifier queue complete"
