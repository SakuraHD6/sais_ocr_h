#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import json


def count_boxes(label_dir: Path) -> tuple[int, int, int]:
    files = sorted(label_dir.glob("*.txt"))
    boxes = 0
    bad = 0
    for path in files:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) != 5:
                bad += 1
                continue
            try:
                vals = [float(v) for v in parts[1:]]
            except ValueError:
                bad += 1
                continue
            if any(v < -1e-6 or v > 1 + 1e-6 for v in vals):
                bad += 1
            boxes += 1
    return len(files), boxes, bad


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("/home/admin/Sais_ocr/work/yolo_data_fixed_20260608"))
    args = parser.parse_args()
    report = {}
    for split in ["train", "val"]:
        image_dir = args.data / "images" / split
        label_dir = args.data / "labels" / split
        image_count = len(list(image_dir.glob("*.png")))
        label_count, boxes, bad = count_boxes(label_dir)
        report[split] = {
            "images": image_count,
            "label_files": label_count,
            "boxes": boxes,
            "bad_lines": bad,
        }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if any(v["bad_lines"] for v in report.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

