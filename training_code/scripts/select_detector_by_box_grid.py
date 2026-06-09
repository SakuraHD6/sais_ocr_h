#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path("/home/admin/Sais_ocr")
SCRIPT_DIR = Path(__file__).resolve().parent
GRID_SCRIPT = SCRIPT_DIR / "run_detector_box_grid.py"

RESUME_B1_WEIGHTS_DIR = (
    "runs/detect/runs/detect/fixed_data_20260608/"
    "yolo11l_full_fixed_e32_resume_b1_from_epoch9_best/weights"
)

DEFAULT_CANDIDATES = [
    (
        "e32_epoch9_best",
        "work/snapshots/yolo11l_e32_epoch9_best_before_resume.pt",
    ),
    ("resume_b1_best", f"{RESUME_B1_WEIGHTS_DIR}/best.pt"),
    ("resume_b1_last", f"{RESUME_B1_WEIGHTS_DIR}/last.pt"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate",
        action="append",
        nargs=2,
        metavar=("NAME", "CHECKPOINT"),
        help="Candidate detector checkpoint. Can be repeated.",
    )
    parser.add_argument("--out", type=Path, default=Path("work/snapshots/yolo11l_selected_detector_best.pt"))
    parser.add_argument("--report", type=Path, default=Path("work/evals/yolo11l_selected_detector_best.json"))
    parser.add_argument("--summary", type=Path, default=Path("work/evals/yolo11l_selected_detector_box_summary.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("work/evals"))
    parser.add_argument("--tag", default="yolo11l_selected_detector_box")
    parser.add_argument("--confs", default="0.08,0.10,0.12,0.15,0.18,0.20,0.25,0.30")
    parser.add_argument("--nms-ious", default="0.60,0.70")
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument("--imgsz", type=int, default=1536)
    parser.add_argument("--max-det", type=int, default=500)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return ROOT / path


def default_candidates_with_epoch_checkpoints() -> list[tuple[str, str]]:
    candidates = list(DEFAULT_CANDIDATES)
    weights_dir = resolve(RESUME_B1_WEIGHTS_DIR)
    for checkpoint in sorted(weights_dir.glob("epoch*.pt")):
        candidates.append((f"resume_b1_{checkpoint.stem}", str(checkpoint)))
    return candidates


def dedupe_candidates(candidates: list[tuple[str, str]]) -> list[tuple[str, str]]:
    deduped: list[tuple[str, str]] = []
    seen: set[Path] = set()
    for name, raw_checkpoint in candidates:
        checkpoint = resolve(raw_checkpoint)
        key = checkpoint.resolve(strict=False)
        if key in seen:
            continue
        seen.add(key)
        deduped.append((name, str(checkpoint)))
    return deduped


def run_grid(args: argparse.Namespace, name: str, checkpoint: Path) -> list[dict]:
    candidate_tag = f"{args.tag}_{name}"
    command = [
        sys.executable,
        str(GRID_SCRIPT),
        "--model",
        str(checkpoint),
        "--tag",
        candidate_tag,
        "--out-dir",
        str(args.out_dir),
        "--confs",
        args.confs,
        "--nms-ious",
        args.nms_ious,
        "--match-iou",
        str(args.match_iou),
        "--imgsz",
        str(args.imgsz),
        "--max-det",
        str(args.max_det),
        "--device",
        args.device,
        "--batch",
        str(args.batch),
    ]
    if args.half:
        command.append("--half")
    if args.overwrite:
        command.append("--overwrite")

    subprocess.run(command, cwd=ROOT, check=True)
    summary_path = args.out_dir / f"{candidate_tag}_summary.json"
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise TypeError(f"unexpected detector grid summary format: {summary_path}")

    rows = []
    for row in data:
        item = dict(row)
        item["candidate_name"] = name
        item["candidate_checkpoint"] = str(checkpoint)
        item["candidate_summary"] = str(summary_path)
        rows.append(item)
    return rows


def selection_key(row: dict) -> tuple[float, float, float, int]:
    return (
        float(row.get("f1") or 0.0),
        float(row.get("recall") or 0.0),
        float(row.get("precision") or 0.0),
        int(row.get("tp") or 0),
    )


def main() -> None:
    args = parse_args()
    args.out_dir = resolve(args.out_dir)
    args.out = resolve(args.out)
    args.report = resolve(args.report)
    args.summary = resolve(args.summary)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    candidates = args.candidate or default_candidates_with_epoch_checkpoints()
    candidates = dedupe_candidates(candidates)
    evaluated: list[dict] = []
    skipped: list[dict] = []
    all_rows: list[dict] = []

    for name, raw_checkpoint in candidates:
        checkpoint = resolve(raw_checkpoint)
        if not checkpoint.exists() or checkpoint.stat().st_size <= 0:
            skipped.append(
                {
                    "name": name,
                    "checkpoint": str(checkpoint),
                    "reason": "missing_or_empty",
                }
            )
            continue
        rows = run_grid(args, name, checkpoint)
        top = max(rows, key=selection_key) if rows else None
        evaluated.append(
            {
                "name": name,
                "checkpoint": str(checkpoint),
                "top_box_summary": top,
                "rows": len(rows),
            }
        )
        all_rows.extend(rows)

    if not all_rows:
        raise SystemExit("no detector box-grid rows were produced")

    all_rows.sort(key=selection_key, reverse=True)
    selected = all_rows[0]
    selected_checkpoint = Path(str(selected["candidate_checkpoint"]))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(selected_checkpoint, args.out)

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(all_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    report = {
        "selection_metric": "box_grid_f1",
        "selection_tiebreakers": ["recall", "precision", "tp"],
        "selected": {
            "name": selected["candidate_name"],
            "checkpoint": str(selected_checkpoint),
            "output_checkpoint": str(args.out),
            "box_summary": selected,
        },
        "summary": str(args.summary),
        "evaluated": evaluated,
        "skipped": skipped,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
