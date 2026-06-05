"""
YOLO detection wrapper. Detects ancient character regions on rubbing images.
"""
import os
import sys
import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO


def _silent_open_image(image_path):
    """Open image suppressing C-level libpng ICC profile warnings on stderr."""
    null = os.devnull
    old_stderr = os.dup(2)
    fd = os.open(null, os.O_WRONLY)
    os.dup2(fd, 2)
    os.close(fd)
    try:
        img = Image.open(image_path)
    finally:
        os.dup2(old_stderr, 2)
        os.close(old_stderr)
    return img


def nms(predictions, iou_threshold=0.5):
    """Non-maximum suppression on predicted bboxes (xywh format)."""
    if not predictions:
        return []

    boxes = np.array([p['bbox'] for p in predictions])
    scores = np.array([p['confidence'] for p in predictions])

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]

    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h

        area_i = (x2[i] - x1[i]) * (y2[i] - y1[i])
        area_o = (x2[order[1:]] - x1[order[1:]]) * (y2[order[1:]] - y1[order[1:]])
        union = area_i + area_o - inter
        iou = inter / np.maximum(union, 1e-10)

        order = order[1:][iou <= iou_threshold]

    return [predictions[i] for i in keep]


def filter_predictions(predictions, img_width=None, img_height=None,
                       min_size=10, conf_threshold=0.5, nms_threshold=0.5):
    """Post-process OCR predictions: filter & deduplicate.

    Args:
        predictions: list of dicts with 'bbox' [x,y,w,h] and 'confidence'
        img_width, img_height: image dimensions (None = skip boundary check)
        min_size: minimum width/height in pixels
        conf_threshold: minimum confidence score
        nms_threshold: IoU threshold for NMS (<=0 to skip NMS)

    Returns:
        Filtered list of predictions.
    """
    filtered = []

    for pred in predictions:
        bbox = pred['bbox']
        x, y, w, h = bbox
        # Confidence filter
        if pred.get('confidence', 1) < conf_threshold:
            continue
        # Size filter
        if w < min_size or h < min_size:
            continue
        # Boundary filter
        if img_width is not None and (x < 0 or y < 0 or x + w > img_width):
            continue
        if img_height is not None and y + h > img_height:
            continue

        filtered.append(pred)

    # NMS
    if nms_threshold > 0 and len(filtered) > 1:
        filtered = nms(filtered, nms_threshold)

    return filtered


class YOLODetector:
    def __init__(self, weights_path, device="cuda:0", conf=0.25,
                 half=False, max_det=300, imgsz=1280, crop_padding=0.15):
        self.model = YOLO(weights_path)
        self.device = device
        self.conf = conf
        self.half = half
        self.max_det = max_det
        self.imgsz = imgsz
        self.crop_padding = crop_padding

    def detect_and_crop(self, image_path):
        """Run detection and return cropped character regions."""
        results = self.model(
            image_path,
            conf=self.conf,
            device=self.device,
            verbose=False,
            half=self.half,
            max_det=self.max_det,
            imgsz=self.imgsz,
        )
        crops = []
        if len(results) == 0:
            return crops

        result = results[0]
        img = _silent_open_image(image_path).convert("RGB")

        if result.boxes is None:
            return crops

        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()

        for box, score in zip(boxes, scores):
            raw_x1, raw_y1, raw_x2, raw_y2 = [int(round(v)) for v in box]
            raw_x1, raw_y1 = max(0, raw_x1), max(0, raw_y1)
            raw_x2, raw_y2 = min(img.width, raw_x2), min(img.height, raw_y2)
            if raw_x2 <= raw_x1 or raw_y2 <= raw_y1:
                continue

            # Keep the submitted bbox tight; use padding only for the classifier crop.
            pad_w = int((raw_x2 - raw_x1) * self.crop_padding)
            pad_h = int((raw_y2 - raw_y1) * self.crop_padding)
            crop_x1 = max(0, raw_x1 - pad_w)
            crop_y1 = max(0, raw_y1 - pad_h)
            crop_x2 = min(img.width, raw_x2 + pad_w)
            crop_y2 = min(img.height, raw_y2 + pad_h)

            crop = img.crop((crop_x1, crop_y1, crop_x2, crop_y2))
            crops.append({
                "bbox": [raw_x1, raw_y1, raw_x2 - raw_x1, raw_y2 - raw_y1],
                "image": crop,
                "confidence": float(score),
            })

        return crops
