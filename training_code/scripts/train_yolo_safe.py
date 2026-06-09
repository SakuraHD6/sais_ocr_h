#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", default="yolo11l.pt")
    parser.add_argument("--project", default="runs/detect/fixed_data_20260608")
    parser.add_argument("--name", default="yolo11l_full_fixed_e1_smoke")
    parser.add_argument("--imgsz", type=int, default=1536)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--optimizer", default="AdamW")
    parser.add_argument("--lr0", type=float, default=0.0001)
    parser.add_argument("--lrf", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--save-period", type=int, default=1)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--check-finite", action="store_true")
    parser.add_argument("--plots", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    return parser.parse_args()


def check_results_finite(run_dir: Path) -> None:
    results = run_dir / "results.csv"
    if not results.exists():
        raise RuntimeError(f"missing results.csv: {results}")
    with results.open("r", encoding="utf-8", newline="") as f:
        for row_index, row in enumerate(csv.DictReader(f), start=2):
            for key, value in row.items():
                value = (value or "").strip()
                if not value:
                    continue
                try:
                    number = float(value)
                except ValueError:
                    continue
                if not math.isfinite(number):
                    raise RuntimeError(f"non-finite value in {results}:{row_index} {key}={value}")


def main() -> None:
    args = parse_args()
    model = YOLO(args.model)
    results = model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        weight_decay=args.weight_decay,
        patience=args.patience,
        save_period=args.save_period,
        fraction=args.fraction,
        exist_ok=args.exist_ok,
        amp=False,
        warmup_epochs=0.0,
        warmup_bias_lr=0.0,
        mosaic=0.0,
        fliplr=0.0,
        close_mosaic=0,
        seed=42,
        deterministic=args.deterministic,
        plots=args.plots,
    )
    save_dir = getattr(results, "save_dir", None)
    run_dir = Path(save_dir) if save_dir is not None else Path(args.project) / args.name
    if args.check_finite:
        check_results_finite(run_dir)


if __name__ == "__main__":
    main()
