#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sais_ocr.ocr_common import collect_image_xml_pairs, dump_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, default=Path("/home/admin/Sais_ocr/dataset/train"))
    parser.add_argument("--out", type=Path, default=Path("/home/admin/Sais_ocr/work/splits/fixed_val280_seed42.json"))
    parser.add_argument("--val-count", type=int, default=280)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pairs, png_unpaired, xml_unpaired = collect_image_xml_pairs(args.src)
    paired = [stem for stem, _, _ in pairs]
    rng = random.Random(args.seed)
    shuffled = paired[:]
    rng.shuffle(shuffled)
    val = sorted(shuffled[: args.val_count])
    train = sorted(shuffled[args.val_count :])
    split = {
        "src": str(args.src),
        "seed": args.seed,
        "val_count": args.val_count,
        "paired_count": len(paired),
        "train": train,
        "val": val,
        "png_without_xml": png_unpaired,
        "xml_without_png": xml_unpaired,
    }
    dump_json(args.out, split)
    print(f"wrote {args.out}")
    print(f"train={len(train)} val={len(val)} paired={len(paired)}")


if __name__ == "__main__":
    main()
