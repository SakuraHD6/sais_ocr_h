#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sais_ocr.ocr_common import Box, collect_image_xml_pairs, dump_json, parse_xml_records


def expand_box(box: Box, image_size: tuple[int, int], padding: float, jitter: tuple[float, float] = (0.0, 0.0)) -> tuple[int, int, int, int]:
    width, height = image_size
    bw = box.width
    bh = box.height
    dx = bw * padding
    dy = bh * padding
    jx, jy = jitter
    x1 = max(0, int(round(box.x1 - dx + jx * bw)))
    y1 = max(0, int(round(box.y1 - dy + jy * bh)))
    x2 = min(width, int(round(box.x2 + dx + jx * bw)))
    y2 = min(height, int(round(box.y2 + dy + jy * bh)))
    if x2 <= x1:
        x2 = min(width, x1 + 1)
    if y2 <= y1:
        y2 = min(height, y1 + 1)
    return x1, y1, x2, y2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, default=Path("/home/admin/Sais_ocr/dataset/train"))
    parser.add_argument("--split", type=Path, default=Path("/home/admin/Sais_ocr/work/splits/fixed_val280_seed42.json"))
    parser.add_argument("--out", type=Path, default=Path("/home/admin/Sais_ocr/work/classifier_crops_fixed_20260608"))
    parser.add_argument("--train-variants", default="pad00,pad05,pad10,pad15,jitter")
    parser.add_argument("--val-variants", default="pad00,pad10")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-box-size", type=float, default=1.0)
    parser.add_argument("--limit-images", type=int, default=0)
    args = parser.parse_args()

    split = json.loads(args.split.read_text(encoding="utf-8"))
    pair_list, _, _ = collect_image_xml_pairs(args.src)
    pair_map = {stem: (image_path, xml_path) for stem, image_path, xml_path in pair_list}
    rng = random.Random(args.seed)

    train_variants = [v.strip() for v in args.train_variants.split(",") if v.strip()]
    val_variants = [v.strip() for v in args.val_variants.split(",") if v.strip()]
    variant_padding = {
        "pad00": 0.00,
        "pad05": 0.05,
        "pad10": 0.10,
        "pad15": 0.15,
        "pad20": 0.20,
        "jitter": 0.10,
    }

    labels = set()
    all_split_records = {}
    for split_name, stems, variants in [
        ("train", split["train"], train_variants),
        ("val", split["val"], val_variants),
    ]:
        stems = list(stems)
        if args.limit_images > 0:
            stems = stems[: args.limit_images]
        written = 0
        failed = 0
        raw_records = 0
        for stem in tqdm(stems, desc=f"prepare classifier {split_name}"):
            image_path, xml_path = pair_map[stem]
            records = parse_xml_records(xml_path, image_path, min_box_size=args.min_box_size)
            raw_records += len(records)
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                for index, rec in enumerate(records):
                    labels.add(rec.label)
                    for variant in variants:
                        padding = variant_padding[variant]
                        jitter = (0.0, 0.0)
                        if variant == "jitter":
                            jitter = (rng.uniform(-0.08, 0.08), rng.uniform(-0.08, 0.08))
                        crop_box = expand_box(rec.bbox, rec.image_size, padding, jitter)
                        crop = image.crop(crop_box)
                        label_dir = args.out / split_name / rec.label
                        label_dir.mkdir(parents=True, exist_ok=True)
                        crop_path = label_dir / f"{stem}_{index:04d}_{variant}.png"
                        try:
                            crop.save(crop_path, format="PNG", optimize=False, compress_level=1)
                            written += 1
                        except Exception:
                            failed += 1
        all_split_records[split_name] = {
            "images": len(stems),
            "raw_records": raw_records,
            "written": written,
            "failed": failed,
            "variants": variants,
        }

    labels_sorted = sorted(labels)
    class_mapping = {str(i): label for i, label in enumerate(labels_sorted)}
    char_to_id = {label: i for i, label in enumerate(labels_sorted)}
    dump_json(args.out / "class_mapping.json", class_mapping)
    dump_json(args.out / "char_to_id.json", char_to_id)
    stats = {
        "src": str(args.src),
        "split": str(args.split),
        "out": str(args.out),
        "labels": len(labels_sorted),
        "splits": all_split_records,
    }
    dump_json(args.out / "stats.json", stats)
    print(f"wrote {args.out}")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

