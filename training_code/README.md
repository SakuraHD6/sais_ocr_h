# Training Code Audit Notes

This directory is reserved for semifinal code audit materials required by the
competition organizer.

For the final-week semifinal image, place complete and reproducible preprocessing,
training, evaluation, and inference code under `/app/training_code`, and keep the
whole training-code package under 5 GB.

The online evaluator will not run this training code. The container entrypoint
still runs `/app/run.sh` for inference and writes `/saisresult/prediction.json`.

Current inference assets copied into the image:

- `/app/src/`
- `/app/yolo_dataset/weights/best.pt`
- `/app/classifier_output/best.pth`
- `/app/class_mapping.json`

Runtime constraints to preserve:

- Do not perform network calls in inference code.
- Do not install packages at container runtime.
- Read test images from `/saisdata/50/eval/images`.
- Write only `/saisresult/prediction.json` for predictions.
