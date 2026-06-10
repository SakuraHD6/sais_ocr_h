#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path("/home/admin/Sais_ocr")
SUBMISSION_DIR = ROOT / "submit/sais_ocr_h_69"
DEFAULT_SUMMARIES = [
    ROOT / "work/evals/pipeline_effb0_128_fixed_summary.json",
    ROOT / "work/evals/pipeline_tf_effv2s_224_fixed_summary.json",
]
DEFAULT_CLASS_MAPPING = ROOT / "work/classifier_crops_fixed_20260608/class_mapping.json"
DEFAULT_REPORT = ROOT / "work/evals/sync_h69_from_best_pipeline_report.json"


def now() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_summary_rows(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [dict(row) for row in data]
    if isinstance(data, dict) and "summary" in data:
        return [dict(data["summary"])]
    if isinstance(data, dict):
        return [dict(data)]
    raise TypeError(f"unsupported summary format: {path}")


def as_float(value: Any, default: float = -1.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def best_from_summaries(summary_paths: list[Path]) -> tuple[dict[str, Any] | None, list[dict[str, str]]]:
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for path in summary_paths:
        path = resolve(path)
        if not path.exists() or path.stat().st_size <= 0:
            skipped.append({"summary": rel(path), "reason": "missing_or_empty"})
            continue
        try:
            loaded = load_summary_rows(path)
        except Exception as exc:
            skipped.append({"summary": rel(path), "reason": repr(exc)})
            continue
        for index, row in enumerate(loaded):
            row["summary_path"] = rel(path)
            row["summary_index"] = index
            rows.append(row)
    rows.sort(key=lambda row: as_float(row.get("e2e_f1")), reverse=True)
    return (rows[0] if rows else None), skipped


def current_h69_e2e_f1(manifest: Path) -> float:
    if not manifest.exists():
        return -1.0
    text = manifest.read_text(encoding="utf-8")
    patterns = [
        r"E2E F1:\s*`?([0-9.]+)`?",
        r"e2e_f1:\s*`?([0-9.]+)`?",
    ]
    values: list[float] = []
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            values.append(as_float(match.group(1)))
    return max(values) if values else -1.0


def copy_file(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def env_value(row: dict[str, Any], key: str, default: Any) -> str:
    value = row.get(key, default)
    if value is None:
        value = default
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


def replace_env_value(dockerfile: Path, key: str, value: str) -> None:
    text = dockerfile.read_text(encoding="utf-8")
    pattern = re.compile(rf"(^\s*{re.escape(key)}=)([^\s\\]+)(\s*\\?\s*$)", re.MULTILINE)
    replacement = rf"\g<1>{value}\g<3>"
    new_text, count = pattern.subn(replacement, text, count=1)
    if count == 0:
        raise RuntimeError(f"could not find ENV key in Dockerfile: {key}")
    dockerfile.write_text(new_text, encoding="utf-8")


def update_docker_env(submit_dir: Path, row: dict[str, Any]) -> None:
    dockerfile = submit_dir / "Dockerfile"
    updates = {
        "CONFIDENCE_THRESHOLD": env_value(row, "det_conf", "0.30"),
        "YOLO_IOU_THRESHOLD": env_value(row, "det_iou", "0.70"),
        "YOLO_IMGSZ": env_value(row, "imgsz", "1536"),
        "MAX_DET": env_value(row, "max_det", "600"),
        "CROP_PADDING": env_value(row, "crop_padding", "0.05"),
        "CLASSIFIER_BATCH": env_value(row, "cls_batch", "96"),
        "HALF": "1" if bool(row.get("det_half", True)) else "0",
        "NMS_IOU_THRESHOLD": env_value(row, "det_iou", "0.70"),
    }
    for key, value in updates.items():
        replace_env_value(dockerfile, key, value)


def write_manifest(
    submit_dir: Path,
    row: dict[str, Any],
    detector_hash: str,
    classifier_hash: str,
    mapping_hash: str,
    previous_f1: float,
) -> None:
    manifest = submit_dir / "SUBMISSION_MANIFEST.md"
    manifest.write_text(
        "\n".join(
            [
                "# sais_ocr_h_69",
                "",
                f"Synced at: {now()}",
                "",
                "This submission directory is based on `submit/sais_ocr_h` and keeps the Dockerfile package installation flow aligned with that template. The package is synced only from full-pipeline validation evidence.",
                "",
                "## Model Selection",
                "",
                f"- Detector: `{row.get('det_weights')}`",
                f"- Classifier: `{row.get('cls_weights')}`",
                "- Class mapping: `work/classifier_crops_fixed_20260608/class_mapping.json`",
                f"- Source summary: `{row.get('summary_path', 'unknown')}`",
                f"- Previous h69 E2E F1 used for gate: `{previous_f1}`",
                "",
                "## Local Fixed-GT Validation",
                "",
                f"- E2E F1: `{row.get('e2e_f1')}`",
                f"- Box F1: `{row.get('box_f1')}`",
                f"- Matched text accuracy: `{row.get('text_accuracy_on_box_match')}`",
                f"- E2E precision / recall: `{row.get('e2e_precision')}` / `{row.get('e2e_recall')}`",
                f"- Box precision / recall: `{row.get('box_precision')}` / `{row.get('box_recall')}`",
                "",
                "## Runtime Defaults",
                "",
                f"- `CONFIDENCE_THRESHOLD={row.get('det_conf')}`",
                f"- `YOLO_IOU_THRESHOLD={row.get('det_iou')}`",
                f"- `YOLO_IMGSZ={row.get('imgsz')}`",
                f"- `MAX_DET={row.get('max_det')}`",
                f"- `CROP_PADDING={row.get('crop_padding')}`",
                f"- `CLASSIFIER_BATCH={row.get('cls_batch')}`",
                f"- `HALF={'1' if bool(row.get('det_half', True)) else '0'}`",
                "",
                "## Weight Hashes",
                "",
                f"- Detector SHA256: `{detector_hash}`",
                f"- Classifier SHA256: `{classifier_hash}`",
                f"- Class mapping SHA256: `{mapping_hash}`",
                "",
                "## Submission Requirement Notes",
                "",
                "Checked against `competition_docx/round2_submission_update_2026-06-05.md`:",
                "",
                "- Runtime inference writes `/saisresult/prediction.json`.",
                "- Prediction bbox format is `[x, y, w, h]`.",
                "- Runtime entrypoint does not install packages or download files.",
                "- Dockerfile installation flow is kept aligned with the original `submit/sais_ocr_h` template.",
                "- Lightweight audit material is included under `/app/training_code`.",
                "- Full training data, crop caches, large logs, and extra checkpoints are not included.",
                "",
                "## Notes",
                "",
                "- `src/run_inference.py` uses the rebuild classifier checkpoint format.",
                "- Prediction bbox output is `[x, y, w, h]`, matching the original submission template documentation.",
                "- `training_code/eval_records/` contains local fixed-GT evidence for the selected detector/classifier pair.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", action="append", type=Path, help="Pipeline summary JSON. Can be repeated.")
    parser.add_argument("--submit-dir", type=Path, default=SUBMISSION_DIR)
    parser.add_argument("--class-mapping", type=Path, default=DEFAULT_CLASS_MAPPING)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--min-delta", type=float, default=1e-9)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="Sync even if selected F1 is not better than current h69.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    submit_dir = resolve(args.submit_dir)
    summary_paths = args.summary or DEFAULT_SUMMARIES
    selected, skipped = best_from_summaries(summary_paths)
    current_f1 = current_h69_e2e_f1(submit_dir / "SUBMISSION_MANIFEST.md")
    selected_f1 = as_float(selected.get("e2e_f1")) if selected else -1.0
    should_sync = bool(selected and (args.force or selected_f1 > current_f1 + args.min_delta))

    report: dict[str, Any] = {
        "generated_at": now(),
        "submit_dir": rel(submit_dir),
        "current_h69_e2e_f1": current_f1,
        "selected_e2e_f1": selected_f1,
        "selected": selected,
        "skipped": skipped,
        "dry_run": args.dry_run,
        "synced": False,
    }

    if not selected:
        report["reason"] = "no_valid_pipeline_summary"
    elif not should_sync:
        report["reason"] = "selected_not_better_than_current_h69"
    elif args.dry_run:
        report["reason"] = "dry_run_selected_better"
    else:
        det_weights = resolve(Path(str(selected["det_weights"])))
        cls_weights = resolve(Path(str(selected["cls_weights"])))
        class_mapping = resolve(args.class_mapping)
        copy_file(det_weights, submit_dir / "yolo_dataset/detect_yolo11l/weights/best.pt")
        copy_file(cls_weights, submit_dir / "classifier_output/best.pth")
        copy_file(cls_weights, submit_dir / "classifier_output/best_infer.pth")
        copy_file(class_mapping, submit_dir / "class_mapping.json")
        update_docker_env(submit_dir, selected)
        detector_hash = sha256(submit_dir / "yolo_dataset/detect_yolo11l/weights/best.pt")
        classifier_hash = sha256(submit_dir / "classifier_output/best.pth")
        mapping_hash = sha256(submit_dir / "class_mapping.json")
        write_manifest(submit_dir, selected, detector_hash, classifier_hash, mapping_hash, current_f1)
        report.update(
            {
                "synced": True,
                "reason": "selected_pipeline_better_than_current_h69",
                "detector_sha256": detector_hash,
                "classifier_sha256": classifier_hash,
                "class_mapping_sha256": mapping_hash,
            }
        )

    report_path = resolve(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
