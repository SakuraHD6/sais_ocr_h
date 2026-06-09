#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image
from tqdm import tqdm


def clean_one(src: Path, dst: Path, compress_level: int) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as image:
        image.convert("RGB").save(dst, format="PNG", optimize=False, compress_level=compress_level)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("/home/admin/Sais_ocr/work/yolo_data_fixed_20260608"))
    parser.add_argument("--out", type=Path, default=Path("/home/admin/Sais_ocr/work/yolo_data_fixed_20260608_clean"))
    parser.add_argument("--compress-level", type=int, default=0)
    args = parser.parse_args()

    for split in ["train", "val"]:
        src_image_dir = args.data / "images" / split
        dst_image_dir = args.out / "images" / split
        src_label_dir = args.data / "labels" / split
        dst_label_dir = args.out / "labels" / split
        dst_label_dir.mkdir(parents=True, exist_ok=True)
        for label in src_label_dir.glob("*.txt"):
            target = dst_label_dir / label.name
            if target.exists() or target.is_symlink():
                target.unlink()
            target.symlink_to(label.resolve())
        for image in tqdm(sorted(src_image_dir.glob("*.png")), desc=f"clean {split}"):
            target = dst_image_dir / image.name
            clean_one(image.resolve(), target, args.compress_level)
    data_yaml = args.out / "data.yaml"
    data_yaml.write_text(
        f"path: {args.out.resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: char\n",
        encoding="utf-8",
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

