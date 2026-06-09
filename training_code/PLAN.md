# SAIS OCR Competition Push Plan

## Summary

The repository was cleaned and now only `dataset/`, `competition_docx/`, and this
new rebuild workspace remain. Old code, weights, virtualenvs, and train outputs
cannot be used directly. The first priority is to rebuild a corrected evaluation
and data pipeline, then train detector/classifier models under one fixed GT
contract.

All new scores must use the corrected XML parser. Historical scores from before
the coordinate fix are directional only.

## Current Facts

- `dataset/train`: `6236` PNG files and `6237` XML files.
- `dataset/HUST-OBC`: auxiliary OCR character data, not the primary SAIS GT.
- No tmux training session is currently running.
- Disk has enough room for rebuild outputs.
- Historical best reliable direction:
  - old optimized detector/classifier baseline around local F1 `0.516`.
  - hybrid/prototype direction improved substantially, but old `0.585022` was
    from a pre-fix GT mouth and must be recomputed.

## Execution Order

1. Rebuild minimal Python environment and install dependencies.
2. Implement shared corrected XML parser and bbox utilities.
3. Create a fixed train/val split with `seed=42`, `val=280`.
4. Generate cleaned image copies and corrected YOLO full-image labels.
5. Run parser/data smoke checks.
6. Train detector:
   - `e1 smoke`
   - `e32 safe`
   - `e150 long` only if e32 direction is useful
7. Generate full classifier crops and train:
   - `efficientnet_b0@128`
   - `tf_efficientnetv2_s@224`
8. Rebuild hybrid/prototype inference and rerank grid.
9. Build a submission package only after fixed-GT full-pipeline F1 improves.

## Score Gates

- Detector mAP alone is not enough.
- A candidate must improve fixed-GT full-pipeline F1.
- Strong candidate target: `F1 >= 0.60`.
- Do not submit a model that only increases TP by adding many FP.

## Output Layout

- Code: `code/sais_ocr_rebuild_20260608`
- Logs: `work/logs`
- Splits: `work/splits`
- Generated detector data: `work/yolo_data_fixed_20260608`
- Generated classifier data: `work/classifier_crops_fixed_20260608`
- YOLO runs: `runs/detect/fixed_data_20260608`
- Classifier runs: `runs/classifier`
- Submission candidates: `local/`

