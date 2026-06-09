#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
EVAL_SCRIPT = SCRIPT_DIR / "eval_pipeline.py"


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--det-weights", required=True)
    parser.add_argument("--cls-weights", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("/home/admin/Sais_ocr/work/evals"))
    parser.add_argument("--confs", default="0.15,0.18,0.20,0.22,0.25,0.30")
    parser.add_argument("--nms-ious", default="0.70")
    parser.add_argument("--crop-paddings", default="0.05,0.10,0.15")
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument("--imgsz", type=int, default=1536)
    parser.add_argument("--max-det", type=int, default=600)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--det-batch", type=int, default=2)
    parser.add_argument("--det-half", action="store_true")
    parser.add_argument("--cls-batch", type=int, default=256)
    parser.add_argument("--limit-images", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def run_eval(args: argparse.Namespace, conf: float, nms_iou: float, crop_padding: float, out_file: Path) -> dict:
    command = [
        sys.executable,
        str(EVAL_SCRIPT),
        "--det-weights",
        args.det_weights,
        "--cls-weights",
        args.cls_weights,
        "--out",
        str(out_file),
        "--imgsz",
        str(args.imgsz),
        "--det-conf",
        str(conf),
        "--det-iou",
        str(nms_iou),
        "--match-iou",
        str(args.match_iou),
        "--max-det",
        str(args.max_det),
        "--crop-padding",
        str(crop_padding),
        "--device",
        args.device,
        "--det-batch",
        str(args.det_batch),
        "--cls-batch",
        str(args.cls_batch),
    ]
    if args.det_half:
        command.append("--det-half")
    if args.limit_images > 0:
        command.extend(["--limit-images", str(args.limit_images)])
    subprocess.run(command, check=True)
    data = json.loads(out_file.read_text(encoding="utf-8"))
    return data["summary"]


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    confs = parse_float_list(args.confs)
    nms_ious = parse_float_list(args.nms_ious)
    crop_paddings = parse_float_list(args.crop_paddings)

    summaries: list[dict] = []
    for conf in confs:
        for nms_iou in nms_ious:
            for crop_padding in crop_paddings:
                suffix = f"conf{conf:.3f}_nms{nms_iou:.2f}_pad{crop_padding:.2f}".replace(".", "p")
                out_file = args.out_dir / f"{args.tag}_{suffix}.json"
                if out_file.exists() and not args.overwrite:
                    data = json.loads(out_file.read_text(encoding="utf-8"))
                    summary = data["summary"]
                else:
                    summary = run_eval(args, conf, nms_iou, crop_padding, out_file)
                summaries.append(summary)

    summaries.sort(key=lambda item: item["e2e_f1"], reverse=True)
    summary_json = args.out_dir / f"{args.tag}_summary.json"
    summary_csv = args.out_dir / f"{args.tag}_summary.csv"
    summary_json.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    if summaries:
        fieldnames = [
            "det_weights",
            "cls_weights",
            "images",
            "gt",
            "pred",
            "det_conf",
            "det_iou",
            "match_iou",
            "imgsz",
            "max_det",
            "crop_padding",
            "det_batch",
            "det_half",
            "cls_batch",
            "classifier_classes",
            "gt_label_coverage",
            "box_tp",
            "box_fp",
            "box_fn",
            "box_precision",
            "box_recall",
            "box_f1",
            "text_correct_on_box_match",
            "text_accuracy_on_box_match",
            "e2e_tp",
            "e2e_fp",
            "e2e_fn",
            "e2e_precision",
            "e2e_recall",
            "e2e_f1",
        ]
        with summary_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(summaries)
        print(json.dumps(summaries[0], ensure_ascii=False, indent=2))
        print(f"wrote {summary_json}")
        print(f"wrote {summary_csv}")


if __name__ == "__main__":
    main()
