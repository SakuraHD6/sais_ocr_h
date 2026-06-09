# Training Code README

This directory contains lightweight semifinal audit material for the current
`sais_ocr_h_69` submission package. It is copied into the image as
`/app/training_code`.

The online evaluator does not run these files. The container entrypoint still
runs `/app/run.sh` and writes `/saisresult/prediction.json`.

## Environment

- Python: system `python3`
- CUDA/PyTorch used during rebuild: see `ENVIRONMENT.md`
- Required Python packages: see `requirements.txt`
- Future package downloads should use the Tsinghua PyPI mirror when needed.

## Data

Official training data is expected at:

```text
/home/admin/Sais_ocr/dataset
```

Large data directories are intentionally not included in this Docker context.
See `data_manifest.md` for the expected source paths and regenerated output
paths.

## Training And Preprocessing Code

Copied code:

```text
scripts/
  prepare_yolo_dataset.py
  clean_yolo_images.py
  train_yolo_safe.py
  prepare_classifier_crops.py
  train_classifier.py
  eval_classifier.py
  eval_detector_boxes.py
  eval_pipeline.py
  run_pipeline_grid.py
  select_detector_checkpoint.py
  select_detector_by_box_grid.py
  select_pipeline_candidate.py
  sync_h69_from_best_pipeline.py
  monitor_tf_pipeline_then_sync_h69.sh
sais_ocr/
  ocr_common.py
```

The selected submission uses:

```text
detector:   work/snapshots/yolo11l_selected_detector_best.pt
classifier: runs/classifier/effb0_128_fixed/best.pth
mapping:    work/classifier_crops_fixed_20260608/class_mapping.json
```

The larger TF-EffV2S classifier was still training when this package was
prepared and is not used here because no full-pipeline validation result was
available yet.

## Key Commands

Prepare YOLO data:

```bash
python3 scripts/prepare_yolo_dataset.py --help
```

Train detector:

```bash
python3 scripts/train_yolo_safe.py --help
```

Prepare classifier crops:

```bash
python3 scripts/prepare_classifier_crops.py --help
```

Train classifier:

```bash
python3 scripts/train_classifier.py --help
```

Run full pipeline validation:

```bash
python3 scripts/run_pipeline_grid.py --help
```

## Inference

Runtime inference code is outside this audit directory:

```text
/app/src/run_inference.py
```

Important environment variables:

```text
INPUT_DIR=/saisdata/50/eval/images
OUTPUT_FILE=/saisresult/prediction.json
DETECTION_WEIGHTS=/app/yolo_dataset/weights/best.pt
CLASSIFIER_WEIGHTS=/app/classifier_output/best.pth
CLASS_MAPPING=/app/class_mapping.json
DEVICE=cuda:0
```

Prediction output is UTF-8 JSON. Each image id maps to a list of predictions,
with `bbox` in `[x, y, w, h]` format and `text` as a string.

## Validation Evidence

`eval_records/` contains the local fixed-GT evidence for the selected current
best:

```text
E2E F1:    0.32375690607734803
Box F1:    0.6556169429097606
Text acc:  0.49157303370786515
```

Runtime constraints to preserve:

- Do not perform network calls in inference code.
- Do not install packages at container runtime.
- Read test images from `/saisdata/50/eval/images`.
- Write only `/saisresult/prediction.json` for predictions.
- Keep this audit package lightweight; do not include full datasets, crop
  caches, large logs, or extra checkpoints.
