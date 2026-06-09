#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

from PIL import Image
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sais_ocr.ocr_common import (
    collect_image_xml_pairs,
    dump_json,
    parse_xml_records,
    yolo_line_from_box,
)


def save_clean_png(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as image:
        image.convert("RGB").save(dst, format="PNG", optimize=False, compress_level=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, default=Path("/home/admin/Sais_ocr/dataset/train"))
    parser.add_argument("--split", type=Path, default=Path("/home/admin/Sais_ocr/work/splits/fixed_val280_seed42.json"))
    parser.add_argument("--out", type=Path, default=Path("/home/admin/Sais_ocr/work/yolo_data_fixed_20260608"))
    parser.add_argument("--copy-mode", choices=["symlink", "clean", "copy"], default="symlink")
    parser.add_argument("--min-box-size", type=float, default=1.0)
    args = parser.parse_args()

    split = json.loads(args.split.read_text(encoding="utf-8"))
    train_stems = set(split["train"])
    val_stems = set(split["val"])
    pairs, _, _ = collect_image_xml_pairs(args.src)
    pair_map = {stem: (image_path, xml_path) for stem, image_path, xml_path in pairs}

    all_records = []
    labels = set()
    per_split_stats = {}
    for split_name, stems in [("train", sorted(train_stems)), ("val", sorted(val_stems))]:
        image_dir = args.out / "images" / split_name
        label_dir = args.out / "labels" / split_name
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        split_records = 0
        missing = []
        for stem in tqdm(stems, desc=f"prepare yolo {split_name}"):
            if stem not in pair_map:
                missing.append(stem)
                continue
            image_path, xml_path = pair_map[stem]
            records = parse_xml_records(xml_path, image_path, min_box_size=args.min_box_size)
            for rec in records:
                labels.add(rec.label)
            all_records.extend(records)

            out_image = image_dir / f"{stem}.png"
            out_label = label_dir / f"{stem}.txt"
            if out_image.exists() or out_image.is_symlink():
                out_image.unlink()
            if args.copy_mode == "symlink":
                out_image.symlink_to(image_path.resolve())
            elif args.copy_mode == "clean":
                save_clean_png(image_path, out_image)
            else:
                shutil.copy2(image_path, out_image)
            lines = [
                yolo_line_from_box(rec.bbox, rec.image_size, 0)
                for rec in records
            ]
            out_label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            split_records += len(records)

        per_split_stats[split_name] = {
            "images": len(stems),
            "boxes": split_records,
            "missing_pairs": missing,
        }

    data_yaml = {
        "path": str(args.out.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "char"},
    }
    with (args.out / "data.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f, allow_unicode=True, sort_keys=False)

    stats = {
        "src": str(args.src),
        "split": str(args.split),
        "out": str(args.out),
        "copy_mode": args.copy_mode,
        "min_box_size": args.min_box_size,
        "labels": len(labels),
        "total_boxes": len(all_records),
        "splits": per_split_stats,
    }
    dump_json(args.out / "stats.json", stats)
    print(f"wrote {args.out}")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
