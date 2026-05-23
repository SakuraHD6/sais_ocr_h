#!/usr/bin/env python3
"""
Inference pipeline: YOLO detection → classification.
Reads images from INPUT_DIR, outputs prediction.json to OUTPUT_FILE.
"""
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

# Suppress libpng ICCP warnings from dataset images
os.environ.setdefault("PIL_LOG_LEVEL", "ERROR")

# Fix random seeds for reproducibility (required by competition rules)
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Add parent dir for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from src.detection_model import YOLODetector, filter_predictions
from src.classification_model import CharacterClassifier


INPUT_DIR = Path(os.getenv("INPUT_DIR", "/saisdata/13/eval/images"))
OUTPUT_FILE = Path(os.getenv("OUTPUT_FILE", "/saisresult/prediction.json"))
DETECTION_WEIGHTS = os.getenv("DETECTION_WEIGHTS", "/app/yolo_dataset/weights/best.pt")
CLASSIFIER_WEIGHTS = os.getenv("CLASSIFIER_WEIGHTS", "/app/classifier_output/best.pth")
ID_TO_CHAR_MAPPING = os.getenv("ID_TO_CHAR_MAPPING", "/app/char_mapping.json")
DEVICE = os.getenv("DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.45"))
HALF = os.getenv("HALF", "1") not in {"0", "false", "False", "no"}
MAX_DET = int(os.getenv("MAX_DET", "100"))
YOLO_IMGSZ = int(os.getenv("YOLO_IMGSZ", "1280"))
MIN_BOX_SIZE = int(os.getenv("MIN_BOX_SIZE", "10"))
NMS_IOU_THRESHOLD = float(os.getenv("NMS_IOU_THRESHOLD", "0.45"))
POST_CONFIDENCE_THRESHOLD = float(os.getenv("POST_CONFIDENCE_THRESHOLD", "0.0"))
MAX_OUTPUT_PER_IMAGE = int(os.getenv("MAX_OUTPUT_PER_IMAGE", "0"))


def find_images():
    suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    if INPUT_DIR.exists():
        return sorted(
            p for p in INPUT_DIR.iterdir()
            if p.suffix.lower() in suffixes
        )
    # Fallback: search recursively
    fallback = Path("/saisdata")
    if fallback.exists():
        return sorted(
            p for p in fallback.rglob("*")
            if p.suffix.lower() in suffixes
        )
    return []


def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(DEVICE)
    print(f"Device: {device}")

    # Load models
    detector = YOLODetector(DETECTION_WEIGHTS, device, conf=CONFIDENCE_THRESHOLD,
                            half=HALF, max_det=MAX_DET, imgsz=YOLO_IMGSZ)
    classifier = CharacterClassifier(CLASSIFIER_WEIGHTS, device, id_to_char_path=ID_TO_CHAR_MAPPING)

    image_paths = find_images()
    print(f"Images found: {len(image_paths)}")

    if not image_paths:
        with OUTPUT_FILE.open("w", encoding="utf-8") as f:
            json.dump({}, f, ensure_ascii=False, indent=2)
        print(f"Empty result saved: {OUTPUT_FILE}")
        return

    results = {}
    for idx, img_path in enumerate(image_paths, 1):
        if idx == 1 or idx % 50 == 0:
            print(f"[{idx}/{len(image_paths)}] {img_path.name}")

        image_id = img_path.stem
        try:
            # Step 1: Detect character regions with YOLO
            crops = detector.detect_and_crop(str(img_path))

            # Step 2: Classify each crop
            predictions = []
            for crop in crops:
                bbox = crop["bbox"]
                char_img = crop["image"]
                char_id, rec_conf = classifier.predict(char_img)
                det_conf = crop["confidence"]
                predictions.append({
                    "bbox": bbox,
                    "text": char_id,
                    "confidence": float(det_conf),
                    "recognition_confidence": float(rec_conf),
                })

            predictions = filter_predictions(
                predictions,
                min_size=MIN_BOX_SIZE,
                conf_threshold=POST_CONFIDENCE_THRESHOLD,
                nms_threshold=NMS_IOU_THRESHOLD,
            )
            if MAX_OUTPUT_PER_IMAGE > 0 and len(predictions) > MAX_OUTPUT_PER_IMAGE:
                predictions = sorted(
                    predictions,
                    key=lambda item: item.get("confidence", 0.0),
                    reverse=True,
                )[:MAX_OUTPUT_PER_IMAGE]
            predictions = sorted(predictions, key=lambda item: (item["bbox"][1], item["bbox"][0]))

            results[image_id] = [
                {
                    "bbox": [int(v) for v in pred["bbox"]],
                    "text": str(pred["text"]),
                }
                for pred in predictions
            ]
        except Exception as e:
            print(f"Warning: failed to process {img_path}: {e}")
            results[image_id] = []

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved: {OUTPUT_FILE}")
    print(f"Total images processed: {len(results)}")


if __name__ == "__main__":
    main()
