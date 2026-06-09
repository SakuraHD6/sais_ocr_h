#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--img-dir", type=Path, required=True)
    parser.add_argument("--require-all-images", action="store_true")
    parser.add_argument("--max-errors", type=int, default=20)
    return parser.parse_args()


def image_map(img_dir: Path) -> dict[str, Path]:
    if not img_dir.exists():
        raise FileNotFoundError(img_dir)
    return {
        path.stem: path
        for path in sorted(img_dir.iterdir())
        if path.suffix.lower() in IMAGE_SUFFIXES
    }


def main() -> None:
    args = parse_args()
    data = json.loads(args.pred.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit("prediction root must be an object")

    images = image_map(args.img_dir)
    errors: list[str] = []
    total_items = 0
    total_empty = 0

    if args.require_all_images:
        missing = sorted(set(images) - set(data))
        for image_id in missing[: args.max_errors]:
            errors.append(f"missing image id: {image_id}")

    for image_id, items in data.items():
        if image_id not in images:
            errors.append(f"unknown image id: {image_id}")
            if len(errors) >= args.max_errors:
                break
            continue
        if not isinstance(items, list):
            errors.append(f"{image_id}: value must be a list")
            if len(errors) >= args.max_errors:
                break
            continue
        if not items:
            total_empty += 1
        with Image.open(images[image_id]) as image:
            width, height = image.size
        for index, item in enumerate(items):
            total_items += 1
            if not isinstance(item, dict):
                errors.append(f"{image_id}[{index}]: item must be an object")
                continue
            bbox = item.get("bbox")
            text = item.get("text")
            if not isinstance(text, str):
                errors.append(f"{image_id}[{index}].text must be a string")
            if not isinstance(bbox, list) or len(bbox) != 4:
                errors.append(f"{image_id}[{index}].bbox must be a 4-item list")
                continue
            if not all(isinstance(value, int) for value in bbox):
                errors.append(f"{image_id}[{index}].bbox values must be ints: {bbox}")
                continue
            x1, y1, x2, y2 = bbox
            if x2 <= x1 or y2 <= y1:
                errors.append(f"{image_id}[{index}].bbox has non-positive area: {bbox}")
            if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
                errors.append(f"{image_id}[{index}].bbox outside image {width}x{height}: {bbox}")
            if len(errors) >= args.max_errors:
                break
        if len(errors) >= args.max_errors:
            break

    summary = {
        "prediction": str(args.pred),
        "images": len(images),
        "prediction_images": len(data),
        "items": total_items,
        "empty_images": total_empty,
        "errors": errors,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
