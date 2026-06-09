#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sais_ocr.ocr_common import collect_image_xml_pairs, dump_json, parse_xml_records, read_image_size


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, default=Path("/home/admin/Sais_ocr/dataset/train"))
    parser.add_argument("--out", type=Path, default=Path("/home/admin/Sais_ocr/work/reports/dataset_summary.json"))
    args = parser.parse_args()

    xml_files = sorted(args.src.glob("*.xml"))
    png_files = sorted(args.src.glob("*.png"))
    pairs, png_unpaired, xml_unpaired = collect_image_xml_pairs(args.src)

    labels: Counter[str] = Counter()
    position_parts: Counter[str] = Counter()
    records_per_image: Counter[int] = Counter()
    mismatched_sizes = 0
    parse_errors: list[dict] = []
    total_records = 0

    for stem, image_path, xml_path in pairs:
        try:
            image_size = read_image_size(image_path)
            records = parse_xml_records(xml_path, image_path)
        except Exception as exc:
            parse_errors.append({"stem": stem, "error": str(exc)})
            continue
        total_records += len(records)
        records_per_image[len(records)] += 1
        for rec in records:
            labels[rec.label] += 1
            raw = rec.raw_position.replace(";", ",")
            count = len([p for p in raw.split(",") if p.strip()])
            position_parts[str(count)] += 1
            if rec.xml_size != image_size:
                mismatched_sizes += 1

    summary = {
        "src": str(args.src),
        "xml_files": len(xml_files),
        "png_files": len(png_files),
        "paired_images": len(pairs),
        "xml_without_png": len(xml_unpaired),
        "png_without_xml": len(png_unpaired),
        "xml_without_png_examples": xml_unpaired[:50],
        "png_without_xml_examples": png_unpaired[:50],
        "valid_records": total_records,
        "labels": len(labels),
        "top_labels": labels.most_common(30),
        "position_part_counts": dict(sorted(position_parts.items(), key=lambda x: int(x[0]))),
        "records_per_image_top": records_per_image.most_common(20),
        "records_with_mismatched_xml_image_size": mismatched_sizes,
        "parse_errors": parse_errors[:50],
        "parse_error_count": len(parse_errors),
    }
    dump_json(args.out, summary)
    print(f"wrote {args.out}")
    print(f"paired_images={len(pairs)} valid_records={total_records} labels={len(labels)} errors={len(parse_errors)}")


if __name__ == "__main__":
    main()
