#!/usr/bin/env python3
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import sys

from PIL import Image
from tqdm import tqdm
import torch
from torchvision import transforms
from ultralytics import YOLO
import timm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sais_ocr.ocr_common import Box, bbox_iou, collect_image_xml_pairs, dump_json, parse_xml_records


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


def open_image_silent(path: Path) -> Image.Image:
    with suppress_stderr():
        image = Image.open(path)
        image.load()
    return image


@dataclass
class LabeledBox:
    bbox: Box
    label: str
    confidence: float = 1.0
    cls_confidence: float = 1.0


@dataclass
class ImageSummary:
    image_id: str
    gt: int
    pred: int
    box_tp: int
    box_fp: int
    box_fn: int
    text_correct_on_box_match: int
    e2e_tp: int
    e2e_fp: int
    e2e_fn: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--det-weights", required=True)
    parser.add_argument("--cls-weights", required=True)
    parser.add_argument("--src", type=Path, default=Path("/home/admin/Sais_ocr/dataset/train"))
    parser.add_argument("--split", type=Path, default=Path("/home/admin/Sais_ocr/work/splits/fixed_val280_seed42.json"))
    parser.add_argument("--split-name", choices=["train", "val"], default="val")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=1536)
    parser.add_argument("--det-conf", type=float, default=0.20)
    parser.add_argument("--det-iou", type=float, default=0.70)
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument("--max-det", type=int, default=600)
    parser.add_argument("--crop-padding", type=float, default=0.10)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--det-batch", type=int, default=2)
    parser.add_argument("--det-half", action="store_true")
    parser.add_argument(
        "--fallback-imgsz",
        default="1280,1024,768",
        help="Comma-separated lower imgsz values retried when a large image OOMs.",
    )
    parser.add_argument("--cls-batch", type=int, default=256)
    parser.add_argument("--limit-images", type=int, default=0)
    parser.add_argument("--min-box-size", type=float, default=1.0)
    parser.add_argument("--save-predictions", action="store_true")
    return parser.parse_args()


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def f1_score(precision: float, recall: float) -> float:
    denom = precision + recall
    return 2.0 * precision * recall / denom if denom else 0.0


def load_split_stems(path: Path, split_name: str) -> list[str]:
    split = json.loads(path.read_text(encoding="utf-8"))
    stems = split[split_name]
    if not isinstance(stems, list):
        raise TypeError(f"{path}:{split_name} must be a list")
    return [str(stem) for stem in stems]


def build_transform(img_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def load_classifier(path: str, device: torch.device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model_name = checkpoint["model_name"]
    img_size = int(checkpoint["img_size"])
    idx_to_class = {int(k): str(v) for k, v in checkpoint["idx_to_class"].items()}
    model = timm.create_model(model_name, pretrained=False, num_classes=len(idx_to_class))
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return model, build_transform(img_size), idx_to_class


def result_boxes(result) -> list[dict]:
    if result.boxes is None or len(result.boxes) == 0:
        return []
    xyxy = result.boxes.xyxy.detach().cpu().numpy()
    confs = result.boxes.conf.detach().cpu().numpy()
    boxes = []
    for coords, conf in zip(xyxy, confs):
        x1, y1, x2, y2 = [float(v) for v in coords.tolist()]
        box = Box(min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
        if box.area > 0:
            boxes.append({"bbox": box, "confidence": float(conf)})
    return boxes


def is_cuda_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return isinstance(exc, torch.cuda.OutOfMemoryError) or (
        "cuda" in text and "out of memory" in text
    )


def detector_imgsz_attempts(primary: int, fallback: str) -> list[int]:
    values = [primary]
    for value in parse_int_list(fallback):
        if value > 0 and value not in values:
            values.append(value)
    return values


def predict_one_image(detector: YOLO, image_path: str, args: argparse.Namespace, device: torch.device, imgsz_values: list[int]):
    for imgsz in imgsz_values:
        try:
            with suppress_stderr():
                results = detector.predict(
                    source=image_path,
                    imgsz=imgsz,
                    conf=args.det_conf,
                    iou=args.det_iou,
                    max_det=args.max_det,
                    device=str(device),
                    batch=1,
                    half=args.det_half,
                    stream=False,
                    verbose=False,
                )
            return results[0], imgsz, False
        except Exception as exc:
            if not is_cuda_oom(exc):
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(
                f"warning: CUDA OOM for {image_path} at imgsz={imgsz}; retrying lower imgsz",
                file=sys.stderr,
                flush=True,
            )
    print(
        f"warning: CUDA OOM for {image_path} at all imgsz attempts; using empty detections",
        file=sys.stderr,
        flush=True,
    )
    return None, 0, True


def expand_box(box: Box, image_size: tuple[int, int], padding: float) -> tuple[int, int, int, int]:
    width, height = image_size
    dx = box.width * padding
    dy = box.height * padding
    x1 = max(0, int(round(box.x1 - dx)))
    y1 = max(0, int(round(box.y1 - dy)))
    x2 = min(width, int(round(box.x2 + dx)))
    y2 = min(height, int(round(box.y2 + dy)))
    if x2 <= x1:
        x2 = min(width, x1 + 1)
    if y2 <= y1:
        y2 = min(height, y1 + 1)
    return x1, y1, x2, y2


def classify_boxes(
    image_path: Path,
    det_boxes: list[dict],
    classifier,
    transform,
    idx_to_class: dict[int, str],
    device: torch.device,
    batch_size: int,
    padding: float,
) -> list[LabeledBox]:
    if not det_boxes:
        return []
    labeled: list[LabeledBox] = []
    with open_image_silent(image_path) as image:
        image = image.convert("RGB")
        image_size = image.size
        crops = []
        meta = []
        for det in det_boxes:
            crop = image.crop(expand_box(det["bbox"], image_size, padding))
            crops.append(transform(crop))
            meta.append(det)

    for start in range(0, len(crops), batch_size):
        batch = torch.stack(crops[start : start + batch_size]).to(device, non_blocking=True)
        with torch.no_grad():
            probs = classifier(batch).softmax(dim=1)
            confs, preds = probs.max(dim=1)
        for offset, (pred, cls_conf) in enumerate(zip(preds.cpu().tolist(), confs.cpu().tolist())):
            det = meta[start + offset]
            labeled.append(
                LabeledBox(
                    bbox=det["bbox"],
                    label=idx_to_class[int(pred)],
                    confidence=float(det["confidence"]),
                    cls_confidence=float(cls_conf),
                )
            )
    return labeled


def greedy_match(
    gt_boxes: list[LabeledBox],
    pred_boxes: list[LabeledBox],
    match_iou: float,
    *,
    require_text: bool,
) -> tuple[int, list[tuple[int, int]]]:
    candidates = []
    for pred_index, pred in enumerate(pred_boxes):
        for gt_index, gt in enumerate(gt_boxes):
            if require_text and pred.label != gt.label:
                continue
            iou = bbox_iou(pred.bbox, gt.bbox)
            if iou >= match_iou:
                candidates.append((iou, pred.confidence, pred_index, gt_index))
    candidates.sort(reverse=True, key=lambda item: (item[0], item[1]))

    used_preds: set[int] = set()
    used_gt: set[int] = set()
    matches: list[tuple[int, int]] = []
    for _, _, pred_index, gt_index in candidates:
        if pred_index in used_preds or gt_index in used_gt:
            continue
        used_preds.add(pred_index)
        used_gt.add(gt_index)
        matches.append((pred_index, gt_index))
    return len(matches), matches


def main() -> None:
    args = parse_args()
    stems = load_split_stems(args.split, args.split_name)
    if args.limit_images > 0:
        stems = stems[: args.limit_images]

    pairs, _, _ = collect_image_xml_pairs(args.src)
    pair_map = {stem: (image_path, xml_path) for stem, image_path, xml_path in pairs}
    image_paths: list[str] = []
    gt_by_stem: dict[str, list[LabeledBox]] = {}
    missing = [stem for stem in stems if stem not in pair_map]
    if missing:
        raise RuntimeError(f"missing image/xml pairs for {len(missing)} stems, first={missing[:3]}")
    for stem in stems:
        image_path, xml_path = pair_map[stem]
        records = parse_xml_records(xml_path, image_path, min_box_size=args.min_box_size)
        gt_by_stem[stem] = [LabeledBox(record.bbox, record.label) for record in records]
        image_paths.append(str(image_path))

    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    detector = YOLO(args.det_weights)
    classifier, cls_transform, idx_to_class = load_classifier(args.cls_weights, device)

    total_gt = total_pred = 0
    total_box_tp = total_box_fp = total_box_fn = 0
    total_e2e_tp = total_e2e_fp = total_e2e_fn = 0
    total_text_correct = 0
    covered_gt = 0
    per_image: list[ImageSummary] = []
    predictions_dump: dict[str, list[dict]] = {}
    covered_labels = set(idx_to_class.values())
    imgsz_values = detector_imgsz_attempts(args.imgsz, args.fallback_imgsz)
    oom_fallbacks = 0
    oom_failures = 0

    for stem, image_path_str in tqdm(zip(stems, image_paths), total=len(stems), desc="eval pipeline"):
        result, used_imgsz, failed_oom = predict_one_image(detector, image_path_str, args, device, imgsz_values)
        if failed_oom:
            oom_failures += 1
        elif used_imgsz != args.imgsz:
            oom_fallbacks += 1
        result_stem = Path(str(getattr(result, "path", image_path_str))).stem if result is not None else Path(image_path_str).stem
        image_id = result_stem if result_stem in gt_by_stem else stem
        image_path, _ = pair_map[image_id]
        gt_boxes = gt_by_stem[image_id]
        det_boxes = result_boxes(result) if result is not None else []
        pred_boxes = classify_boxes(
            image_path,
            det_boxes,
            classifier,
            cls_transform,
            idx_to_class,
            device,
            args.cls_batch,
            args.crop_padding,
        )

        box_tp, box_matches = greedy_match(gt_boxes, pred_boxes, args.match_iou, require_text=False)
        text_correct = sum(
            1
            for pred_index, gt_index in box_matches
            if pred_boxes[pred_index].label == gt_boxes[gt_index].label
        )
        e2e_tp, _ = greedy_match(gt_boxes, pred_boxes, args.match_iou, require_text=True)

        gt_count = len(gt_boxes)
        pred_count = len(pred_boxes)
        box_fp = pred_count - box_tp
        box_fn = gt_count - box_tp
        e2e_fp = pred_count - e2e_tp
        e2e_fn = gt_count - e2e_tp

        total_gt += gt_count
        total_pred += pred_count
        total_box_tp += box_tp
        total_box_fp += box_fp
        total_box_fn += box_fn
        total_e2e_tp += e2e_tp
        total_e2e_fp += e2e_fp
        total_e2e_fn += e2e_fn
        total_text_correct += text_correct
        covered_gt += sum(1 for gt in gt_boxes if gt.label in covered_labels)

        per_image.append(
            ImageSummary(
                image_id=image_id,
                gt=gt_count,
                pred=pred_count,
                box_tp=box_tp,
                box_fp=box_fp,
                box_fn=box_fn,
                text_correct_on_box_match=text_correct,
                e2e_tp=e2e_tp,
                e2e_fp=e2e_fp,
                e2e_fn=e2e_fn,
            )
        )
        if args.save_predictions:
            predictions_dump[image_id] = [
                {
                    "bbox": pred.bbox.to_int_xyxy(),
                    "text": pred.label,
                    "confidence": pred.confidence,
                    "classification_confidence": pred.cls_confidence,
                }
                for pred in pred_boxes
            ]

    box_precision = safe_ratio(total_box_tp, total_box_tp + total_box_fp)
    box_recall = safe_ratio(total_box_tp, total_box_tp + total_box_fn)
    e2e_precision = safe_ratio(total_e2e_tp, total_e2e_tp + total_e2e_fp)
    e2e_recall = safe_ratio(total_e2e_tp, total_e2e_tp + total_e2e_fn)
    summary = {
        "det_weights": args.det_weights,
        "cls_weights": args.cls_weights,
        "images": len(stems),
        "gt": total_gt,
        "pred": total_pred,
        "det_conf": args.det_conf,
        "det_iou": args.det_iou,
        "match_iou": args.match_iou,
        "imgsz": args.imgsz,
        "max_det": args.max_det,
        "crop_padding": args.crop_padding,
        "det_batch": args.det_batch,
        "det_half": args.det_half,
        "fallback_imgsz": imgsz_values[1:],
        "oom_fallbacks": oom_fallbacks,
        "oom_failures": oom_failures,
        "cls_batch": args.cls_batch,
        "classifier_classes": len(idx_to_class),
        "gt_label_coverage": safe_ratio(covered_gt, total_gt),
        "box_tp": total_box_tp,
        "box_fp": total_box_fp,
        "box_fn": total_box_fn,
        "box_precision": box_precision,
        "box_recall": box_recall,
        "box_f1": f1_score(box_precision, box_recall),
        "text_correct_on_box_match": total_text_correct,
        "text_accuracy_on_box_match": safe_ratio(total_text_correct, total_box_tp),
        "e2e_tp": total_e2e_tp,
        "e2e_fp": total_e2e_fp,
        "e2e_fn": total_e2e_fn,
        "e2e_precision": e2e_precision,
        "e2e_recall": e2e_recall,
        "e2e_f1": f1_score(e2e_precision, e2e_recall),
    }
    output = {"summary": summary, "per_image": [asdict(item) for item in per_image]}
    if args.save_predictions:
        output["predictions"] = predictions_dump
    dump_json(args.out, output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
