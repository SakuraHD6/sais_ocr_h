#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil


ROOT = Path("/home/admin/Sais_ocr")
RUN_COMPETITION = ROOT / "code/sais_ocr_rebuild_20260608/scripts/run_competition.py"
REQUIREMENTS = ROOT / "code/sais_ocr_rebuild_20260608/requirements.txt"


DOCKERFILE_TEMPLATE = """FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN python3 -m pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --trusted-host pypi.tuna.tsinghua.edu.cn -r /app/requirements.txt && \
    (python3 -m pip uninstall -y opencv-python opencv-contrib-python opencv-contrib-python-headless || true) && \
    python3 -m pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --trusted-host pypi.tuna.tsinghua.edu.cn opencv-python-headless

COPY run_competition.py /app/run_competition.py
COPY yolo_dataset/weights/best.pt /app/yolo_dataset/weights/best.pt
COPY classifier_output/best.pth /app/classifier_output/best.pth

ENV INPUT_DIR=/saisdata/50/eval/images
ENV OUTPUT_FILE=/saisresult/prediction.json
ENV DETECTION_WEIGHTS=/app/yolo_dataset/weights/best.pt
ENV CLASSIFIER_WEIGHTS=/app/classifier_output/best.pth
ENV DEVICE=cuda:0
ENV CONFIDENCE_THRESHOLD={confidence_threshold}
ENV YOLO_IOU_THRESHOLD={yolo_iou_threshold}
ENV YOLO_IMGSZ={yolo_imgsz}
ENV YOLO_FALLBACK_IMGSZ=1280,1024,768
ENV MAX_DET={max_det}
ENV CROP_PADDING={crop_padding}
ENV CLASSIFIER_BATCH={classifier_batch}
ENV HALF={half}

CMD ["python3", "/app/run_competition.py"]
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True, help="Pipeline grid summary JSON.")
    parser.add_argument("--out", type=Path, required=True, help="Candidate output directory.")
    parser.add_argument("--min-e2e-f1", type=float, default=0.60)
    parser.add_argument("--det-weights", type=Path, default=None)
    parser.add_argument("--cls-weights", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_best_summary(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        if not data:
            raise RuntimeError(f"empty summary list: {path}")
        return data[0]
    if isinstance(data, dict) and "summary" in data:
        return data["summary"]
    if isinstance(data, dict):
        return data
    raise TypeError(f"unsupported summary format: {path}")


def copy_file(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def best_value(best: dict, key: str, default):
    value = best.get(key, default)
    return default if value is None else value


def main() -> None:
    args = parse_args()
    best = load_best_summary(args.summary)
    e2e_f1 = float(best.get("e2e_f1", -1.0))
    if e2e_f1 < args.min_e2e_f1 and not args.force:
        raise SystemExit(
            f"refusing to build candidate: e2e_f1={e2e_f1:.6f} < {args.min_e2e_f1:.6f}; "
            "use --force only for debugging"
        )

    det_weights = args.det_weights or Path(str(best["det_weights"]))
    cls_weights = args.cls_weights or Path(str(best["cls_weights"]))
    if not det_weights.is_absolute():
        det_weights = ROOT / det_weights
    if not cls_weights.is_absolute():
        cls_weights = ROOT / cls_weights

    if args.out.exists():
        if not (args.overwrite or args.force):
            raise SystemExit(f"output exists; pass --overwrite to replace files: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)

    copy_file(RUN_COMPETITION, args.out / "run_competition.py")
    copy_file(REQUIREMENTS, args.out / "requirements.txt")
    copy_file(det_weights, args.out / "yolo_dataset/weights/best.pt")
    copy_file(cls_weights, args.out / "classifier_output/best.pth")
    dockerfile = DOCKERFILE_TEMPLATE.format(
        confidence_threshold=best_value(best, "det_conf", "0.20"),
        yolo_iou_threshold=best_value(best, "det_iou", "0.70"),
        yolo_imgsz=int(best_value(best, "imgsz", 1536)),
        max_det=int(best_value(best, "max_det", 600)),
        crop_padding=best_value(best, "crop_padding", "0.10"),
        classifier_batch=int(best_value(best, "cls_batch", 256)),
        half="1" if bool(best_value(best, "det_half", True)) else "0",
    )
    (args.out / "Dockerfile").write_text(dockerfile, encoding="utf-8")

    manifest = {
        "summary_path": str(args.summary),
        "best_summary": best,
        "det_weights": str(det_weights),
        "cls_weights": str(cls_weights),
        "runtime_env": {
            "CONFIDENCE_THRESHOLD": best_value(best, "det_conf", "0.20"),
            "YOLO_IOU_THRESHOLD": best_value(best, "det_iou", "0.70"),
            "YOLO_IMGSZ": int(best_value(best, "imgsz", 1536)),
            "YOLO_FALLBACK_IMGSZ": "1280,1024,768",
            "MAX_DET": int(best_value(best, "max_det", 600)),
            "CROP_PADDING": best_value(best, "crop_padding", "0.10"),
            "CLASSIFIER_BATCH": int(best_value(best, "cls_batch", 256)),
            "HALF": "1" if bool(best_value(best, "det_half", True)) else "0",
        },
        "min_e2e_f1": args.min_e2e_f1,
        "overwrite": args.overwrite,
        "forced": args.force,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"candidate={args.out}")


if __name__ == "__main__":
    main()
