#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO
import timm


os.environ.setdefault("PIL_LOG_LEVEL", "ERROR")

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


INPUT_DIR = Path(os.getenv("INPUT_DIR", "/saisdata/50/eval/images"))
OUTPUT_FILE = Path(os.getenv("OUTPUT_FILE", "/saisresult/prediction.json"))
DETECTION_WEIGHTS = os.getenv("DETECTION_WEIGHTS", "/app/yolo_dataset/weights/best.pt")
CLASSIFIER_WEIGHTS = os.getenv("CLASSIFIER_WEIGHTS", "/app/classifier_output/best.pth")
DEVICE = os.getenv("DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")

CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.30"))
YOLO_IOU_THRESHOLD = float(os.getenv("YOLO_IOU_THRESHOLD", "0.70"))
YOLO_IMGSZ = int(os.getenv("YOLO_IMGSZ", "1536"))
MAX_DET = int(os.getenv("MAX_DET", "600"))
HALF = os.getenv("HALF", "1") not in {"0", "false", "False", "no"}

CROP_PADDING = float(os.getenv("CROP_PADDING", "0.05"))
CLASSIFIER_BATCH = int(os.getenv("CLASSIFIER_BATCH", "256"))
MIN_BOX_SIZE = int(os.getenv("MIN_BOX_SIZE", "1"))
FINAL_SCORE_THRESHOLD = float(os.getenv("FINAL_SCORE_THRESHOLD", "0.0"))
MAX_OUTPUT_PER_IMAGE = int(os.getenv("MAX_OUTPUT_PER_IMAGE", "0"))


@dataclass
class Detection:
    bbox: tuple[float, float, float, float]
    confidence: float


@dataclass
class Prediction:
    bbox: tuple[float, float, float, float]
    text: str
    det_confidence: float
    cls_confidence: float

    @property
    def score(self) -> float:
        return self.det_confidence * self.cls_confidence


class ClassifierTransform:
    def __init__(self, img_size: int):
        self.img_size = img_size
        self.mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(3, 1, 1)

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB").resize(
            (self.img_size, self.img_size),
            resample=Image.Resampling.BICUBIC,
        )
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array.transpose(2, 0, 1))
        return (tensor - self.mean) / self.std


def find_images() -> list[Path]:
    suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    if INPUT_DIR.exists():
        return sorted(path for path in INPUT_DIR.iterdir() if path.suffix.lower() in suffixes)
    fallback = Path("/saisdata")
    if fallback.exists():
        return sorted(path for path in fallback.rglob("*") if path.suffix.lower() in suffixes)
    return []


def build_transform(img_size: int) -> ClassifierTransform:
    return ClassifierTransform(img_size)


def load_classifier(path: str, device: torch.device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model_name = checkpoint["model_name"]
    img_size = int(checkpoint["img_size"])
    idx_to_class = {int(index): str(label) for index, label in checkpoint["idx_to_class"].items()}
    model = timm.create_model(model_name, pretrained=False, num_classes=len(idx_to_class))
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return model, build_transform(img_size), idx_to_class


def clamp_box(
    bbox: tuple[float, float, float, float],
    image_size: tuple[int, int],
) -> tuple[float, float, float, float] | None:
    width, height = image_size
    x1, y1, x2, y2 = bbox
    x1 = max(0.0, min(float(width), x1))
    y1 = max(0.0, min(float(height), y1))
    x2 = max(0.0, min(float(width), x2))
    y2 = max(0.0, min(float(height), y2))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    if (x2 - x1) < MIN_BOX_SIZE or (y2 - y1) < MIN_BOX_SIZE:
        return None
    return x1, y1, x2, y2


def expand_box(
    bbox: tuple[float, float, float, float],
    image_size: tuple[int, int],
    padding: float,
) -> tuple[int, int, int, int]:
    width, height = image_size
    x1, y1, x2, y2 = bbox
    dx = (x2 - x1) * padding
    dy = (y2 - y1) * padding
    left = max(0, int(round(x1 - dx)))
    top = max(0, int(round(y1 - dy)))
    right = min(width, int(round(x2 + dx)))
    bottom = min(height, int(round(y2 + dy)))
    if right <= left:
        right = min(width, left + 1)
    if bottom <= top:
        bottom = min(height, top + 1)
    return left, top, right, bottom


def detections_from_result(result, image_size: tuple[int, int]) -> list[Detection]:
    if result.boxes is None or len(result.boxes) == 0:
        return []
    xyxy = result.boxes.xyxy.detach().cpu().numpy()
    confs = result.boxes.conf.detach().cpu().numpy()
    detections: list[Detection] = []
    for coords, conf in zip(xyxy, confs):
        bbox = clamp_box(tuple(float(value) for value in coords.tolist()), image_size)
        if bbox is not None:
            detections.append(Detection(bbox=bbox, confidence=float(conf)))
    return detections


def classify_detections(
    image_path: Path,
    detections: list[Detection],
    classifier,
    transform,
    idx_to_class: dict[int, str],
    device: torch.device,
    *,
    batch_size: int,
    crop_padding: float,
    use_half: bool,
) -> list[Prediction]:
    if not detections:
        return []
    predictions: list[Prediction] = []
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image_size = image.size
        crops = [
            transform(image.crop(expand_box(det.bbox, image_size, crop_padding)))
            for det in detections
        ]

    for start in range(0, len(crops), batch_size):
        batch = torch.stack(crops[start : start + batch_size]).to(device, non_blocking=True)
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=use_half and device.type == "cuda"):
                probs = classifier(batch).softmax(dim=1)
            confs, preds = probs.max(dim=1)
        for offset, (pred_id, cls_conf) in enumerate(zip(preds.cpu().tolist(), confs.cpu().tolist())):
            det = detections[start + offset]
            pred = Prediction(
                bbox=det.bbox,
                text=idx_to_class[int(pred_id)],
                det_confidence=det.confidence,
                cls_confidence=float(cls_conf),
            )
            if pred.score >= FINAL_SCORE_THRESHOLD:
                predictions.append(pred)
    return predictions


def format_bbox(bbox: tuple[float, float, float, float]) -> list[int]:
    x1, y1, x2, y2 = bbox
    x = int(round(x1))
    y = int(round(y1))
    w = max(1, int(round(x2 - x1)))
    h = max(1, int(round(y2 - y1)))
    return [x, y, w, h]


def main() -> None:
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(DEVICE if torch.cuda.is_available() or not DEVICE.startswith("cuda") else "cpu")
    use_half = HALF and device.type == "cuda"
    print(f"device={device} half={use_half}")

    image_paths = find_images()
    print(f"images={len(image_paths)} input={INPUT_DIR}")
    if not image_paths:
        OUTPUT_FILE.write_text("{}", encoding="utf-8")
        print(f"saved empty output: {OUTPUT_FILE}")
        return

    detector = YOLO(DETECTION_WEIGHTS)
    classifier, cls_transform, idx_to_class = load_classifier(CLASSIFIER_WEIGHTS, device)
    results: dict[str, list[dict]] = {}

    yolo_results = detector.predict(
        source=[str(path) for path in image_paths],
        imgsz=YOLO_IMGSZ,
        conf=CONFIDENCE_THRESHOLD,
        iou=YOLO_IOU_THRESHOLD,
        max_det=MAX_DET,
        device=str(device),
        half=use_half,
        stream=True,
        verbose=False,
    )

    for index, (fallback_path, result) in enumerate(zip(image_paths, yolo_results), 1):
        if index == 1 or index % 50 == 0:
            print(f"[{index}/{len(image_paths)}] {fallback_path.name}")
        result_path = Path(str(getattr(result, "path", fallback_path)))
        image_path = result_path if result_path.exists() else fallback_path
        image_id = image_path.stem
        try:
            with Image.open(image_path) as image:
                image_size = image.size
            detections = detections_from_result(result, image_size)
            predictions = classify_detections(
                image_path,
                detections,
                classifier,
                cls_transform,
                idx_to_class,
                device,
                batch_size=CLASSIFIER_BATCH,
                crop_padding=CROP_PADDING,
                use_half=use_half,
            )
            if MAX_OUTPUT_PER_IMAGE > 0 and len(predictions) > MAX_OUTPUT_PER_IMAGE:
                predictions = sorted(predictions, key=lambda item: item.score, reverse=True)[:MAX_OUTPUT_PER_IMAGE]
            predictions = sorted(predictions, key=lambda item: (item.bbox[1], item.bbox[0]))
            results[image_id] = [
                {"bbox": format_bbox(pred.bbox), "text": pred.text}
                for pred in predictions
            ]
        except Exception as exc:
            print(f"warning: failed to process {image_path}: {exc}")
            results[image_id] = []

    OUTPUT_FILE.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved={OUTPUT_FILE} images={len(results)}")


if __name__ == "__main__":
    main()
