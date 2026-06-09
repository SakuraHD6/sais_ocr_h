#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path("/home/admin/Sais_ocr")
BUILD_SCRIPT = ROOT / "code/sais_ocr_rebuild_20260608/scripts/build_submission_candidate.py"

DEFAULT_SUMMARIES = [
    ROOT / "work/evals/pipeline_effb0_128_fixed_summary.json",
    ROOT / "work/evals/pipeline_tf_effv2s_224_fixed_summary.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary",
        action="append",
        type=Path,
        help="Pipeline grid summary JSON. Can be repeated.",
    )
    parser.add_argument("--report", type=Path, default=Path("work/evals/pipeline_best_candidate_report.json"))
    parser.add_argument("--selected-summary", type=Path, default=Path("work/evals/pipeline_best_selected_summary.json"))
    parser.add_argument("--out", type=Path, default=Path("local/sais_ocr_rebuild_20260608_candidate"))
    parser.add_argument("--min-e2e-f1", type=float, default=0.60)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--force", action="store_true", help="Allow build script to bypass its F1 gate.")
    parser.add_argument("--no-build", action="store_true")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def load_summary_rows(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [dict(row) for row in data]
    if isinstance(data, dict) and "summary" in data:
        return [dict(data["summary"])]
    if isinstance(data, dict):
        return [dict(data)]
    raise TypeError(f"unsupported summary format: {path}")


def e2e_f1(row: dict[str, Any]) -> float:
    try:
        return float(row.get("e2e_f1", -1.0))
    except Exception:
        return -1.0


def main() -> None:
    args = parse_args()
    summaries = [resolve(path) for path in (args.summary or DEFAULT_SUMMARIES)]
    report_path = resolve(args.report)
    selected_summary_path = resolve(args.selected_summary)
    out_dir = resolve(args.out)

    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for summary_path in summaries:
        if not summary_path.exists() or summary_path.stat().st_size <= 0:
            skipped.append({"summary": str(summary_path), "reason": "missing_or_empty"})
            continue
        try:
            loaded = load_summary_rows(summary_path)
        except Exception as exc:
            skipped.append({"summary": str(summary_path), "reason": repr(exc)})
            continue
        for index, row in enumerate(loaded):
            row = dict(row)
            row["summary_path"] = str(summary_path)
            row["summary_index"] = index
            rows.append(row)

    rows.sort(key=e2e_f1, reverse=True)
    selected = rows[0] if rows else None
    selected_f1 = e2e_f1(selected) if selected else -1.0
    should_build = bool(selected and selected_f1 >= args.min_e2e_f1 and not args.no_build)

    report: dict[str, Any] = {
        "min_e2e_f1": args.min_e2e_f1,
        "selected": selected,
        "top_rows": rows[:20],
        "skipped": skipped,
        "output_dir": str(out_dir),
        "built": False,
    }

    if selected is None:
        report["reason"] = "no_valid_pipeline_summaries"
    elif selected_f1 < args.min_e2e_f1:
        report["reason"] = f"best_e2e_f1_below_gate:{selected_f1:.6f}"
    elif args.no_build:
        report["reason"] = "no_build_requested"
    else:
        selected_summary_path.parent.mkdir(parents=True, exist_ok=True)
        selected_summary_path.write_text(
            json.dumps([selected], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        command = [
            sys.executable,
            str(BUILD_SCRIPT),
            "--summary",
            str(selected_summary_path),
            "--out",
            str(out_dir),
            "--min-e2e-f1",
            str(args.min_e2e_f1),
        ]
        if args.overwrite:
            command.append("--overwrite")
        if args.force:
            command.append("--force")
        subprocess.run(command, cwd=ROOT, check=True)
        report["built"] = True
        report["reason"] = "built_candidate"
        report["selected_summary_path"] = str(selected_summary_path)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
