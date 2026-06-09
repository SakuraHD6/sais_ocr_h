#!/usr/bin/env python3
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import timm


@contextmanager
def suppress_stderr():
    old_stderr = os.dup(2)
    null_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(null_fd, 2)
        yield
    finally:
        os.dup2(old_stderr, 2)
        os.close(old_stderr)
        os.close(null_fd)


def open_image_silent(path: Path):
    from PIL import Image

    with suppress_stderr():
        image = Image.open(path)
        image.load()
    return image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", type=Path, default=Path("/home/admin/Sais_ocr/work/classifier_crops_fixed_20260608/val"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit-errors", type=int, default=200)
    return parser.parse_args()


def build_transform(img_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


class FixedMappingImageDataset(Dataset):
    def __init__(self, root: Path, class_to_idx: dict[str, int], transform: transforms.Compose) -> None:
        self.root = Path(root)
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []
        for label, class_id in sorted(class_to_idx.items(), key=lambda item: item[1]):
            label_dir = self.root / label
            if not label_dir.exists():
                continue
            for path in sorted(label_dir.iterdir()):
                if path.suffix.lower() in IMAGE_SUFFIXES:
                    self.samples.append((path, class_id))
        if not self.samples:
            raise RuntimeError(f"no classifier images found under {self.root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, target = self.samples[index]

        with open_image_silent(path) as image:
            image = image.convert("RGB")
            return self.transform(image), target


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_name = checkpoint["model_name"]
    img_size = int(checkpoint["img_size"])
    idx_to_class = {int(k): v for k, v in checkpoint["idx_to_class"].items()}
    class_to_idx = {label: idx for idx, label in idx_to_class.items()}

    dataset = FixedMappingImageDataset(args.data, class_to_idx, build_transform(img_size))

    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    model = timm.create_model(model_name, pretrained=False, num_classes=len(idx_to_class))
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    total = 0
    top1 = 0
    top5 = 0
    errors = []
    sample_offset = 0
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(images)
            maxk = min(5, logits.shape[1])
            probs = logits.softmax(dim=1)
            confs, preds = probs.topk(maxk, dim=1)
            correct = preds.eq(targets[:, None])
            top1 += int(correct[:, :1].sum().item())
            top5 += int(correct.any(dim=1).sum().item())
            batch_size = targets.numel()
            if len(errors) < args.limit_errors:
                for i in range(batch_size):
                    if bool(correct[i, 0].item()):
                        continue
                    sample_path, true_idx = dataset.samples[sample_offset + i]
                    errors.append(
                        {
                            "path": str(sample_path),
                            "target": idx_to_class[int(true_idx)],
                            "pred": idx_to_class[int(preds[i, 0].item())],
                            "confidence": float(confs[i, 0].item()),
                            "top5": [
                                {
                                    "label": idx_to_class[int(preds[i, j].item())],
                                    "confidence": float(confs[i, j].item()),
                                }
                                for j in range(maxk)
                            ],
                        }
                    )
                    if len(errors) >= args.limit_errors:
                        break
            total += batch_size
            sample_offset += batch_size

    summary = {
        "checkpoint": args.checkpoint,
        "data": str(args.data),
        "model": model_name,
        "img_size": img_size,
        "classes": len(idx_to_class),
        "samples": total,
        "top1": top1 / max(1, total),
        "top5": top5 / max(1, total),
        "errors_kept": len(errors),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "errors": errors}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
