#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys

import torch
from tqdm import tqdm
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sais_ocr.ocr_common import Box, bbox_iou, collect_image_xml_pairs, dump_json, parse_xml_records


@dataclass
class ImageMetrics:
    image_id: str
    gt: int
    pred: int
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="YOLO checkpoint path")
    parser.add_argument("--src", type=Path, default=Path("/home/admin/Sais_ocr/dataset/train"))
    parser.add_argument("--split", type=Path, default=Path("/home/admin/Sais_ocr/work/splits/fixed_val280_seed42.json"))
    parser.add_argument("--split-name", choices=["train", "val"], default="val")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=1536)
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--iou", type=float, default=0.70, help="YOLO NMS IoU")
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument("--max-det", type=int, default=600)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--half", action="store_true")
    parser.add_argument(
        "--fallback-imgsz",
        default="1280,1024,768",
        help="Comma-separated lower imgsz values retried when a large image OOMs.",
    )
    parser.add_argument("--limit-images", type=int, default=0)
    parser.add_argument("--min-box-size", type=float, default=1.0)
    parser.add_argument("--save-predictions", action="store_true")
    return parser.parse_args()


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def load_split_stems(path: Path, split_name: str) -> list[str]:
    split = json.loads(path.read_text(encoding="utf-8"))
    stems = split[split_name]
    if not isinstance(stems, list):
        raise TypeError(f"{path}:{split_name} must be a list")
    return [str(stem) for stem in stems]


def yolo_result_boxes(result) -> list[dict]:
    if result.boxes is None or len(result.boxes) == 0:
        return []
    xyxy = result.boxes.xyxy.detach().cpu().numpy()
    confs = result.boxes.conf.detach().cpu().numpy()
    preds = []
    for coords, conf in zip(xyxy, confs):
        x1, y1, x2, y2 = [float(v) for v in coords.tolist()]
        box = Box(min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
        if box.area <= 0:
            continue
        preds.append({"bbox": box, "confidence": float(conf)})
    return preds


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


def predict_one_image(model: YOLO, image_path: str, args: argparse.Namespace, imgsz_values: list[int]):
    for imgsz in imgsz_values:
        try:
            results = model.predict(
                source=image_path,
                imgsz=imgsz,
                conf=args.conf,
                iou=args.iou,
                max_det=args.max_det,
                device=args.device,
                batch=1,
                half=args.half,
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


def greedy_match(gt_boxes: list[Box], pred_boxes: list[dict], match_iou: float) -> tuple[int, int, int]:
    candidates = []
    for pred_index, pred in enumerate(pred_boxes):
        for gt_index, gt_box in enumerate(gt_boxes):
            iou = bbox_iou(pred["bbox"], gt_box)
            if iou >= match_iou:
                candidates.append((iou, pred_index, gt_index))
    candidates.sort(reverse=True, key=lambda item: item[0])

    used_preds: set[int] = set()
    used_gt: set[int] = set()
    for _, pred_index, gt_index in candidates:
        if pred_index in used_preds or gt_index in used_gt:
            continue
        used_preds.add(pred_index)
        used_gt.add(gt_index)

    tp = len(used_gt)
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp
    return tp, fp, fn


def safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def f1_score(precision: float, recall: float) -> float:
    denom = precision + recall
    return 2.0 * precision * recall / denom if denom else 0.0


def main() -> None:
    args = parse_args()
    stems = load_split_stems(args.split, args.split_name)
    if args.limit_images > 0:
        stems = stems[: args.limit_images]

    pairs, _, _ = collect_image_xml_pairs(args.src)
    pair_map = {stem: (image_path, xml_path) for stem, image_path, xml_path in pairs}
    missing = [stem for stem in stems if stem not in pair_map]
    if missing:
        raise RuntimeError(f"missing image/xml pairs for {len(missing)} stems, first={missing[:3]}")

    gt_by_stem: dict[str, list[Box]] = {}
    image_paths = []
    for stem in stems:
        image_path, xml_path = pair_map[stem]
        records = parse_xml_records(xml_path, image_path, min_box_size=args.min_box_size)
        gt_by_stem[stem] = [record.bbox for record in records]
        image_paths.append(str(image_path))

    model = YOLO(args.model)
    imgsz_values = detector_imgsz_attempts(args.imgsz, args.fallback_imgsz)
    oom_fallbacks = 0
    oom_failures = 0

    total_tp = total_fp = total_fn = 0
    per_image: list[ImageMetrics] = []
    predictions_dump: dict[str, list[dict]] = {}
    for stem, image_path in tqdm(zip(stems, image_paths), total=len(image_paths), desc="eval detector"):
        result, used_imgsz, failed_oom = predict_one_image(model, image_path, args, imgsz_values)
        if failed_oom:
            oom_failures += 1
        elif used_imgsz != args.imgsz:
            oom_fallbacks += 1
        result_stem = Path(str(getattr(result, "path", image_path))).stem if result is not None else Path(image_path).stem
        image_id = result_stem if result_stem in gt_by_stem else stem
        pred_boxes = yolo_result_boxes(result) if result is not None else []
        gt_boxes = gt_by_stem[image_id]
        tp, fp, fn = greedy_match(gt_boxes, pred_boxes, args.match_iou)
        total_tp += tp
        total_fp += fp
        total_fn += fn

        precision = safe_ratio(tp, tp + fp)
        recall = safe_ratio(tp, tp + fn)
        per_image.append(
            ImageMetrics(
                image_id=image_id,
                gt=len(gt_boxes),
                pred=len(pred_boxes),
                tp=tp,
                fp=fp,
                fn=fn,
                precision=precision,
                recall=recall,
                f1=f1_score(precision, recall),
            )
        )
        if args.save_predictions:
            predictions_dump[image_id] = [
                {
                    "bbox": pred["bbox"].to_int_xyxy(),
                    "confidence": pred["confidence"],
                }
                for pred in pred_boxes
            ]

    precision = safe_ratio(total_tp, total_tp + total_fp)
    recall = safe_ratio(total_tp, total_tp + total_fn)
    summary = {
        "model": args.model,
        "src": str(args.src),
        "split": str(args.split),
        "split_name": args.split_name,
        "images": len(image_paths),
        "imgsz": args.imgsz,
        "conf": args.conf,
        "nms_iou": args.iou,
        "match_iou": args.match_iou,
        "max_det": args.max_det,
        "fallback_imgsz": imgsz_values[1:],
        "oom_fallbacks": oom_fallbacks,
        "oom_failures": oom_failures,
        "gt": total_tp + total_fn,
        "pred": total_tp + total_fp,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": precision,
        "recall": recall,
        "f1": f1_score(precision, recall),
    }
    output = {
        "summary": summary,
        "per_image": [asdict(item) for item in per_image],
    }
    if args.save_predictions:
        output["predictions"] = predictions_dump
    dump_json(args.out, output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
