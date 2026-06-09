#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
import timm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("/home/admin/Sais_ocr/work/classifier_crops_fixed_20260608"))
    parser.add_argument("--out", type=Path, default=Path("/home/admin/Sais_ocr/runs/classifier/effb0_128_fixed"))
    parser.add_argument("--model", default="efficientnet_b0")
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--pretrained-fallback", action="store_true")
    parser.add_argument("--weighted-sampler", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-every", type=int, default=5)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_transforms(img_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    train_tf = transforms.Compose(
        [
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomApply([transforms.RandomRotation(5, fill=255)], p=0.25),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.12, contrast=0.12)], p=0.35),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    val_tf = transforms.Compose(
        [
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    return train_tf, val_tf


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
        with Image.open(path) as image:
            image = image.convert("RGB")
            return self.transform(image), target


def load_mapping(data_root: Path) -> tuple[dict[str, int], dict[str, str]]:
    mapping_path = data_root / "class_mapping.json"
    if not mapping_path.exists():
        raise RuntimeError(f"missing class mapping: {mapping_path}")
    idx_to_class_raw = json.loads(mapping_path.read_text(encoding="utf-8"))
    idx_to_class = {str(int(idx)): str(label) for idx, label in idx_to_class_raw.items()}
    class_to_idx = {label: int(idx) for idx, label in idx_to_class.items()}
    return class_to_idx, idx_to_class


def make_sampler(dataset: FixedMappingImageDataset, num_classes: int) -> WeightedRandomSampler:
    counts = np.bincount([target for _, target in dataset.samples], minlength=num_classes)
    weights = [1.0 / max(1, counts[target]) for _, target in dataset.samples]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def accuracy(logits: torch.Tensor, target: torch.Tensor, topk: tuple[int, ...] = (1, 5)) -> dict[int, float]:
    maxk = min(max(topk), logits.shape[1])
    _, pred = logits.topk(maxk, dim=1)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    values: dict[int, float] = {}
    for k in topk:
        k_eff = min(k, logits.shape[1])
        correct_k = correct[:k_eff].reshape(-1).float().sum(0)
        values[k] = float(correct_k.item() / max(1, target.numel()))
    return values


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> dict[str, float]:
    train = optimizer is not None
    model.train(train)
    total_loss = 0.0
    total_seen = 0
    total_top1 = 0.0
    total_top5 = 0.0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            with torch.amp.autocast("cuda", enabled=scaler is not None):
                logits = model(images)
                loss = criterion(logits, targets)
            if train:
                assert optimizer is not None
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        batch_size = targets.numel()
        acc = accuracy(logits.detach(), targets, topk=(1, 5))
        total_loss += float(loss.detach().item()) * batch_size
        total_top1 += acc[1] * batch_size
        total_top5 += acc[5] * batch_size
        total_seen += batch_size

    return {
        "loss": total_loss / max(1, total_seen),
        "top1": total_top1 / max(1, total_seen),
        "top5": total_top5 / max(1, total_seen),
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    args: argparse.Namespace,
    epoch: int,
    best_top1: float,
    class_to_idx: dict[str, int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    idx_to_class = {str(idx): label for label, idx in class_to_idx.items()}
    torch.save(
        {
            "model": model.state_dict(),
            "model_name": args.model,
            "img_size": args.img_size,
            "epoch": epoch,
            "best_top1": best_top1,
            "class_to_idx": class_to_idx,
            "idx_to_class": idx_to_class,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    train_dir = args.data / "train"
    val_dir = args.data / "val"
    if not train_dir.exists() or not val_dir.exists():
        raise RuntimeError(f"missing train/val crop dirs under {args.data}")

    class_to_idx, idx_to_class = load_mapping(args.data)
    train_tf, val_tf = build_transforms(args.img_size)
    train_ds = FixedMappingImageDataset(train_dir, class_to_idx, train_tf)
    val_ds = FixedMappingImageDataset(val_dir, class_to_idx, val_tf)

    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    pin_memory = device.type == "cuda"
    num_classes = len(idx_to_class)
    sampler = make_sampler(train_ds, num_classes) if args.weighted_sampler else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=pin_memory,
        persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=pin_memory,
        persistent_workers=args.workers > 0,
    )

    if args.pretrained and os.getenv("HF_ENDPOINT") is None:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    try:
        model = timm.create_model(args.model, pretrained=args.pretrained, num_classes=num_classes)
        used_pretrained = bool(args.pretrained)
    except Exception:
        if not args.pretrained or not args.pretrained_fallback:
            raise
        print(
            f"pretrained load failed for {args.model}; falling back to random init",
            flush=True,
        )
        model = timm.create_model(args.model, pretrained=False, num_classes=num_classes)
        used_pretrained = False
    model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.02)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    scaler_or_none = scaler if scaler.is_enabled() else None

    (args.out / "class_mapping.json").write_text(json.dumps(idx_to_class, ensure_ascii=False, indent=2), encoding="utf-8")
    config = vars(args).copy()
    config["data"] = str(args.data)
    config["out"] = str(args.out)
    config["classes"] = num_classes
    config["train_images"] = len(train_ds)
    config["val_images"] = len(val_ds)
    config["used_pretrained"] = used_pretrained
    config["hf_endpoint"] = os.getenv("HF_ENDPOINT", "")
    (args.out / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    metrics_path = args.out / "metrics.csv"
    best_top1 = -math.inf
    best_epoch = 0
    bad_epochs = 0
    with metrics_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "lr", "train_loss", "train_top1", "train_top5", "val_loss", "val_top1", "val_top5", "seconds"],
        )
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            started = time.time()
            train_metrics = run_epoch(model, train_loader, criterion, device, optimizer, scaler_or_none)
            val_metrics = run_epoch(model, val_loader, criterion, device)
            scheduler.step()
            row = {
                "epoch": epoch,
                "lr": optimizer.param_groups[0]["lr"],
                "train_loss": train_metrics["loss"],
                "train_top1": train_metrics["top1"],
                "train_top5": train_metrics["top5"],
                "val_loss": val_metrics["loss"],
                "val_top1": val_metrics["top1"],
                "val_top5": val_metrics["top5"],
                "seconds": time.time() - started,
            }
            writer.writerow(row)
            f.flush()
            print(json.dumps(row, ensure_ascii=False), flush=True)

            save_checkpoint(args.out / "last.pth", model, args, epoch, best_top1, class_to_idx)
            if epoch % args.save_every == 0:
                save_checkpoint(args.out / f"epoch{epoch:03d}.pth", model, args, epoch, best_top1, class_to_idx)
            if val_metrics["top1"] > best_top1:
                best_top1 = val_metrics["top1"]
                best_epoch = epoch
                bad_epochs = 0
                save_checkpoint(args.out / "best.pth", model, args, epoch, best_top1, class_to_idx)
            else:
                bad_epochs += 1
                if bad_epochs >= args.patience:
                    print(f"early stop: best_epoch={best_epoch} best_top1={best_top1:.6f}", flush=True)
                    break


if __name__ == "__main__":
    main()
