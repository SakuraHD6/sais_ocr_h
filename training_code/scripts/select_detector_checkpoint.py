#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil


DEFAULT_CANDIDATES = [
    (
        "e32_epoch9_best",
        "work/snapshots/yolo11l_e32_epoch9_best_before_resume.pt",
        "runs/detect/runs/detect/fixed_data_20260608/yolo11l_full_fixed_e32_safe_from_smoke/results.csv",
    ),
    (
        "resume_b1_best",
        "runs/detect/runs/detect/fixed_data_20260608/yolo11l_full_fixed_e32_resume_b1_from_epoch9_best/weights/best.pt",
        "runs/detect/runs/detect/fixed_data_20260608/yolo11l_full_fixed_e32_resume_b1_from_epoch9_best/results.csv",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate",
        action="append",
        nargs=3,
        metavar=("NAME", "CHECKPOINT", "RESULTS_CSV"),
        help="Candidate detector checkpoint and its results.csv. Can be repeated.",
    )
    parser.add_argument("--out", type=Path, default=Path("work/snapshots/yolo11l_selected_detector_best.pt"))
    parser.add_argument("--report", type=Path, default=Path("work/evals/yolo11l_selected_detector_best.json"))
    parser.add_argument("--metric", default="metrics/mAP50-95(B)")
    parser.add_argument("--secondary-metric", default="metrics/mAP50(B)")
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def metric_value(row: dict[str, str], key: str) -> float:
    try:
        return float((row.get(key) or "nan").strip())
    except ValueError:
        return float("nan")


def best_row(rows: list[dict[str, str]], metric: str, secondary_metric: str) -> dict[str, str] | None:
    finite_rows = [
        row
        for row in rows
        if metric_value(row, metric) == metric_value(row, metric)
        and metric_value(row, secondary_metric) == metric_value(row, secondary_metric)
    ]
    if not finite_rows:
        return None
    return max(
        finite_rows,
        key=lambda row: (
            metric_value(row, metric),
            metric_value(row, secondary_metric),
            metric_value(row, "metrics/recall(B)"),
            metric_value(row, "metrics/precision(B)"),
        ),
    )


def main() -> None:
    args = parse_args()
    candidates = args.candidate or DEFAULT_CANDIDATES

    evaluated = []
    for name, checkpoint_raw, results_raw in candidates:
        checkpoint = Path(checkpoint_raw)
        results = Path(results_raw)
        rows = load_rows(results)
        row = best_row(rows, args.metric, args.secondary_metric)
        item = {
            "name": name,
            "checkpoint": str(checkpoint),
            "checkpoint_exists": checkpoint.exists() and checkpoint.stat().st_size > 0,
            "results": str(results),
            "results_rows": len(rows),
            "best_row": row,
            "metric": metric_value(row, args.metric) if row else None,
            "secondary_metric": metric_value(row, args.secondary_metric) if row else None,
        }
        evaluated.append(item)

    usable = [item for item in evaluated if item["checkpoint_exists"] and item["best_row"] is not None]
    if not usable:
        raise SystemExit("no usable detector checkpoint candidates")

    selected = max(
        usable,
        key=lambda item: (
            item["metric"],
            item["secondary_metric"],
            metric_value(item["best_row"], "metrics/recall(B)"),
            metric_value(item["best_row"], "metrics/precision(B)"),
        ),
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(selected["checkpoint"], args.out)

    report = {
        "metric": args.metric,
        "secondary_metric": args.secondary_metric,
        "selected": selected,
        "output_checkpoint": str(args.out),
        "candidates": evaluated,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
