#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--min-box-f1", type=float, default=0.60)
    parser.add_argument("--min-e2e-f1", type=float, default=0.60)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    if isinstance(data, dict) and "summary" in data and isinstance(data["summary"], dict):
        return [data["summary"]]
    if isinstance(data, dict):
        return [data]
    raise TypeError(f"unsupported summary format: {path}")


def score_key(row: dict) -> tuple[float, float]:
    if "e2e_f1" in row:
        return float(row.get("e2e_f1", 0.0)), float(row.get("box_f1", 0.0))
    return float(row.get("box_f1", row.get("f1", 0.0))), float(row.get("box_recall", row.get("recall", 0.0)))


def compact(row: dict) -> dict:
    keys = [
        "det_weights",
        "cls_weights",
        "images",
        "gt",
        "pred",
        "conf",
        "det_conf",
        "nms_iou",
        "det_iou",
        "imgsz",
        "crop_padding",
        "box_precision",
        "box_recall",
        "box_f1",
        "precision",
        "recall",
        "f1",
        "text_accuracy_on_box_match",
        "e2e_precision",
        "e2e_recall",
        "e2e_f1",
    ]
    return {key: row[key] for key in keys if key in row}


def main() -> None:
    args = parse_args()
    rows = sorted(load_rows(args.summary), key=score_key, reverse=True)
    best = rows[0] if rows else {}
    result = {
        "summary": str(args.summary),
        "rows": len(rows),
        "best": compact(best),
        "top": [compact(row) for row in rows[: args.top]],
        "passes_box_gate": float(best.get("box_f1", best.get("f1", 0.0))) >= args.min_box_f1 if best else False,
        "passes_e2e_gate": float(best.get("e2e_f1", 0.0)) >= args.min_e2e_f1 if best and "e2e_f1" in best else False,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
