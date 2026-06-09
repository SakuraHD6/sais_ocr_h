#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import timm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        default=["efficientnet_b0", "tf_efficientnetv2_s"],
    )
    parser.add_argument("--out", type=Path, default=Path("work/logs/classifier_pretrained_prefetch_status.json"))
    parser.add_argument("--hf-endpoint", default="https://hf-mirror.com")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("HF_ENDPOINT", args.hf_endpoint)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    results = []
    for model_name in args.models:
        started = time.time()
        item = {"model": model_name, "ok": False, "seconds": 0.0, "error": ""}
        try:
            # num_classes=0 avoids allocating a task-specific classifier head while
            # still forcing timm/huggingface cache population for the backbone.
            timm.create_model(model_name, pretrained=True, num_classes=0)
            item["ok"] = True
        except Exception as exc:
            item["error"] = repr(exc)
        item["seconds"] = time.time() - started
        results.append(item)
        args.out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(item, ensure_ascii=False), flush=True)

    if not all(item["ok"] for item in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
