#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
EVAL_SCRIPT = SCRIPT_DIR / "eval_detector_boxes.py"


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("/home/admin/Sais_ocr/work/evals"))
    parser.add_argument("--confs", default="0.05,0.08,0.10,0.12,0.15,0.18,0.20,0.25,0.30")
    parser.add_argument("--nms-ious", default="0.70")
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument("--imgsz", type=int, default=1536)
    parser.add_argument("--max-det", type=int, default=600)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--limit-images", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def run_eval(args: argparse.Namespace, conf: float, nms_iou: float, out_file: Path) -> dict:
    command = [
        sys.executable,
        str(EVAL_SCRIPT),
        "--model",
        args.model,
        "--out",
        str(out_file),
        "--imgsz",
        str(args.imgsz),
        "--conf",
        str(conf),
        "--iou",
        str(nms_iou),
        "--match-iou",
        str(args.match_iou),
        "--max-det",
        str(args.max_det),
        "--device",
        args.device,
        "--batch",
        str(args.batch),
    ]
    if args.half:
        command.append("--half")
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

    summaries: list[dict] = []
    for conf in confs:
        for nms_iou in nms_ious:
            suffix = f"conf{conf:.3f}_nms{nms_iou:.2f}".replace(".", "p")
            out_file = args.out_dir / f"{args.tag}_{suffix}.json"
            if out_file.exists() and not args.overwrite:
                data = json.loads(out_file.read_text(encoding="utf-8"))
                summary = data["summary"]
            else:
                summary = run_eval(args, conf, nms_iou, out_file)
            summaries.append(summary)

    summaries.sort(key=lambda item: item["f1"], reverse=True)
    summary_json = args.out_dir / f"{args.tag}_summary.json"
    summary_csv = args.out_dir / f"{args.tag}_summary.csv"
    summary_json.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    if summaries:
        fieldnames = [
            "model",
            "images",
            "imgsz",
            "conf",
            "nms_iou",
            "match_iou",
            "max_det",
            "gt",
            "pred",
            "tp",
            "fp",
            "fn",
            "precision",
            "recall",
            "f1",
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
