# Data Manifest

This submission package does not include the full training dataset or generated
training caches. Those files are large and are not needed for online inference.

## Source Data

Expected local source root:

```text
/home/admin/Sais_ocr/dataset
```

The preserved competition documentation is stored at:

```text
/home/admin/Sais_ocr/competition_docx
```

## Generated Training Outputs

Detector and classifier training outputs were generated outside the submission
directory:

```text
/home/admin/Sais_ocr/work
/home/admin/Sais_ocr/runs
```

Current selected artifacts:

```text
work/snapshots/yolo11l_selected_detector_best.pt
runs/classifier/effb0_128_fixed/best.pth
work/classifier_crops_fixed_20260608/class_mapping.json
```

These selected artifacts are copied into the inference package as:

```text
/app/yolo_dataset/weights/best.pt
/app/classifier_output/best.pth
/app/class_mapping.json
```

## Excluded From Docker Context

The following are intentionally excluded:

```text
full training images
YOLO tile/full image caches
classifier crop caches
large logs
intermediate checkpoints
unselected model weights
```

This keeps `/app/training_code` under the competition recommendation while
still documenting how to regenerate and audit the training pipeline.
