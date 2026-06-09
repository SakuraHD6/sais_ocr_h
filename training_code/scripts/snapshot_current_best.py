#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path("/home/admin/Sais_ocr")
SNAPSHOT_DIR = ROOT / "code/sais_ocr_rebuild_20260608/current_best_20260609_1353"
STATUS_MD = ROOT / "code/sais_ocr_rebuild_20260608/RUN_STATUS.md"

DETECTOR_REPORT = ROOT / "work/evals/yolo11l_selected_detector_best.json"
DETECTOR_BOX_SUMMARY = ROOT / "work/evals/yolo11l_selected_detector_box_summary.json"
DETECTOR_WEIGHTS = ROOT / "work/snapshots/yolo11l_selected_detector_best.pt"

CLASSIFIER_RUNS = {
    "effb0_128_fixed": ROOT / "runs/classifier/effb0_128_fixed",
    "tf_effv2s_224_fixed": ROOT / "runs/classifier/tf_effv2s_224_fixed",
}

EVAL_DIR = ROOT / "work/evals"


def now() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def as_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def csv_float(row: dict[str, str], key: str) -> float | None:
    return as_float(row.get(key))


def fmt(value: Any, digits: int = 5) -> str:
    number = as_float(value)
    if number is None:
        return "null"
    return f"{number:.{digits}f}"


def copy_if_exists(src: Path, dst_name: str | None = None) -> bool:
    if not src.exists():
        return False
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    dst = SNAPSHOT_DIR / (dst_name or src.name)
    shutil.copy2(src, dst)
    return True


def summarize_classifier(name: str, run_dir: Path) -> dict[str, Any]:
    metrics = run_dir / "metrics.csv"
    rows = read_csv_rows(metrics)
    out: dict[str, Any] = {
        "name": name,
        "run_dir": rel(run_dir),
        "metrics": rel(metrics),
        "best_checkpoint": rel(run_dir / "best.pth"),
        "last_checkpoint": rel(run_dir / "last.pth"),
        "epochs_completed": len(rows),
    }
    if rows:
        last = rows[-1]
        best = max(
            rows,
            key=lambda row: csv_float(row, "val_top1")
            if csv_float(row, "val_top1") is not None
            else -1.0,
        )
        out.update(
            {
                "last_epoch": last.get("epoch"),
                "last_val_top1": csv_float(last, "val_top1"),
                "last_val_top5": csv_float(last, "val_top5"),
                "last_val_loss": csv_float(last, "val_loss"),
                "last_train_top1": csv_float(last, "train_top1"),
                "best_epoch": best.get("epoch"),
                "best_val_top1": csv_float(best, "val_top1"),
                "best_val_top5": csv_float(best, "val_top5"),
                "best_val_loss": csv_float(best, "val_loss"),
            }
        )
        copy_if_exists(metrics, f"{name}_metrics.csv")
        copy_if_exists(metrics, f"{name}_metrics_epoch{last.get('epoch')}.csv")
    return out


def best_pipeline_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary in sorted(EVAL_DIR.glob("pipeline_*_summary.json")):
        data = read_json(summary)
        if not isinstance(data, list) or not data:
            continue
        top = data[0]
        if not isinstance(top, dict):
            continue
        item = dict(top)
        item["_summary"] = rel(summary)
        item["_summary_name"] = summary.name
        rows.append(item)
        copy_if_exists(summary)
        csv_summary = summary.with_suffix(".csv")
        copy_if_exists(csv_summary)

    rows.sort(
        key=lambda row: as_float(row.get("e2e_f1")) if as_float(row.get("e2e_f1")) is not None else -1.0,
        reverse=True,
    )
    return rows


def pipeline_config_path(row: dict[str, Any]) -> Path | None:
    summary_name = row.get("_summary_name")
    if not isinstance(summary_name, str) or not summary_name.endswith("_summary.json"):
        return None
    prefix = summary_name[: -len("_summary.json")]
    det_conf = as_float(row.get("det_conf"))
    det_iou = as_float(row.get("det_iou"))
    pad = as_float(row.get("crop_padding"))
    if det_conf is None or det_iou is None or pad is None:
        return None
    name = (
        f"{prefix}_conf{str(f'{det_conf:.3f}').replace('.', 'p')}"
        f"_nms{str(f'{det_iou:.2f}').replace('.', 'p')}"
        f"_pad{str(f'{pad:.2f}').replace('.', 'p')}.json"
    )
    path = EVAL_DIR / name
    return path if path.exists() else None


def command_output(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(
            cmd, cwd=ROOT, text=True, stderr=subprocess.STDOUT
        ).strip()
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


def active_process_summary() -> dict[str, str]:
    ps = command_output(
        [
            "bash",
            "-lc",
            "ps -eo pid,ppid,stat,etime,%cpu,%mem,cmd | "
            "rg 'train_classifier.py|classifier_queue|run_pipeline_grid.py|"
            "eval_pipeline.py|select_pipeline_candidate.py' || true",
        ]
    )
    gpu = command_output(
        [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    gpu_apps = command_output(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    return {"ps": ps, "gpu": gpu, "gpu_apps": gpu_apps}


def write_readme(summary: dict[str, Any]) -> None:
    detector = summary.get("detector", {})
    classifiers = summary.get("classifiers", {})
    best_pipe = summary.get("best_pipeline") or {}
    tf = classifiers.get("tf_effv2s_224_fixed", {})
    effb0 = classifiers.get("effb0_128_fixed", {})

    lines = [
        f"# Current Best Snapshot - refreshed {summary['generated_at']}",
        "",
        "This directory stores the current best small artifacts and pointers while the training queue continues.",
        "Large checkpoint files are not duplicated here.",
        "",
        "## Selected Detector",
        "",
        f"- checkpoint: `{rel(DETECTOR_WEIGHTS)}`",
        f"- source checkpoint: `{detector.get('source_checkpoint', 'unknown')}`",
        f"- precision: `{fmt(detector.get('precision'))}`",
        f"- recall: `{fmt(detector.get('recall'))}`",
        f"- mAP50: `{fmt(detector.get('mAP50'))}`",
        f"- mAP50-95: `{fmt(detector.get('mAP50-95'))}`",
        "",
        "## Current Best Pipeline",
        "",
    ]

    if best_pipe:
        lines.extend(
            [
                f"- summary: `{best_pipe.get('_summary')}`",
                f"- best config file: `{best_pipe.get('_config_file', 'unknown')}`",
                f"- classifier weights: `{best_pipe.get('cls_weights')}`",
                f"- detector conf / nms / crop padding: `{best_pipe.get('det_conf')}` / `{best_pipe.get('det_iou')}` / `{best_pipe.get('crop_padding')}`",
                f"- box precision / recall / F1: `{fmt(best_pipe.get('box_precision'))}` / `{fmt(best_pipe.get('box_recall'))}` / `{fmt(best_pipe.get('box_f1'))}`",
                f"- matched text accuracy: `{fmt(best_pipe.get('text_accuracy_on_box_match'))}`",
                f"- e2e precision / recall / F1: `{fmt(best_pipe.get('e2e_precision'))}` / `{fmt(best_pipe.get('e2e_recall'))}` / `{fmt(best_pipe.get('e2e_f1'))}`",
                "",
            ]
        )
    else:
        lines.extend(["- no completed pipeline summary yet.", ""])

    lines.extend(
        [
            "## Classifier Status",
            "",
            "EffB0 fixed:",
            "",
            f"- checkpoint: `{effb0.get('best_checkpoint')}`",
            f"- epochs completed: `{effb0.get('epochs_completed')}`",
            f"- best epoch: `{effb0.get('best_epoch')}`",
            f"- best val top1 / top5: `{fmt(effb0.get('best_val_top1'))}` / `{fmt(effb0.get('best_val_top5'))}`",
            f"- last epoch val top1 / top5: `{fmt(effb0.get('last_val_top1'))}` / `{fmt(effb0.get('last_val_top5'))}`",
            "",
            "TF-EffV2S fixed:",
            "",
            f"- checkpoint: `{tf.get('best_checkpoint')}`",
            f"- epochs completed: `{tf.get('epochs_completed')}`",
            f"- best epoch: `{tf.get('best_epoch')}`",
            f"- best val top1 / top5: `{fmt(tf.get('best_val_top1'))}` / `{fmt(tf.get('best_val_top5'))}`",
            f"- last epoch val top1 / top5: `{fmt(tf.get('last_val_top1'))}` / `{fmt(tf.get('last_val_top5'))}`",
            "",
            "## Decision",
            "",
            summary.get("decision", ""),
            "",
            "## Copied Small Artifacts",
            "",
        ]
    )
    for path in sorted(p.name for p in SNAPSHOT_DIR.iterdir() if p.is_file() and p.name != "README.md"):
        lines.append(f"- `{path}`")

    (SNAPSHOT_DIR / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_status_once(summary: dict[str, Any]) -> None:
    marker = f"current best snapshot refresh {summary['generated_at']}"
    body = (
        f"\n## {summary['generated_at']} current best snapshot refreshed\n\n"
        f"- snapshot dir: `{rel(SNAPSHOT_DIR)}`\n"
        f"- best pipeline e2e F1: `{fmt((summary.get('best_pipeline') or {}).get('e2e_f1'))}`\n"
        f"- best pipeline box F1: `{fmt((summary.get('best_pipeline') or {}).get('box_f1'))}`\n"
        f"- TF-EffV2S epochs completed: `{summary['classifiers'].get('tf_effv2s_224_fixed', {}).get('epochs_completed')}`\n"
        f"- TF-EffV2S best val top1: `{fmt(summary['classifiers'].get('tf_effv2s_224_fixed', {}).get('best_val_top1'))}`\n"
        f"- note: snapshot refresh only reads files and does not interrupt training.\n"
    )
    text = STATUS_MD.read_text(encoding="utf-8", errors="replace") if STATUS_MD.exists() else ""
    if marker not in text:
        with STATUS_MD.open("a", encoding="utf-8") as f:
            f.write(body)


def build_summary() -> dict[str, Any]:
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

    detector = {}
    report = read_json(DETECTOR_REPORT)
    if isinstance(report, dict):
        selected = report.get("selected") or {}
        box = selected.get("box_summary") or {}
        detector = {
            "checkpoint": rel(DETECTOR_WEIGHTS),
            "source_checkpoint": selected.get("checkpoint") or box.get("candidate_checkpoint"),
            "precision": box.get("precision"),
            "recall": box.get("recall"),
            "mAP50": box.get("mAP50"),
            "mAP50-95": box.get("mAP50-95"),
            "selection_reason": box.get("selection_reason"),
        }
    copy_if_exists(DETECTOR_REPORT)
    copy_if_exists(DETECTOR_BOX_SUMMARY)

    classifiers = {name: summarize_classifier(name, run_dir) for name, run_dir in CLASSIFIER_RUNS.items()}

    for name in CLASSIFIER_RUNS:
        copy_if_exists(EVAL_DIR / f"{name}_eval.json")

    pipeline_rows = best_pipeline_rows()
    best_pipeline = pipeline_rows[0] if pipeline_rows else None
    if best_pipeline:
        config_path = pipeline_config_path(best_pipeline)
        if config_path:
            copy_if_exists(config_path)
            best_pipeline["_config_file"] = rel(config_path)
        else:
            best_pipeline["_config_file"] = None

    effb0_best = classifiers.get("effb0_128_fixed", {}).get("best_val_top1")
    tf_best = classifiers.get("tf_effv2s_224_fixed", {}).get("best_val_top1")
    decision = (
        "Keep the queue running. The selected detector is the current detector best. "
        "EffB0 pipeline is the current completed end-to-end best, while TF-EffV2S is still training."
    )
    if as_float(tf_best) is not None and as_float(effb0_best) is not None:
        if as_float(tf_best) > as_float(effb0_best):
            decision = (
                "TF-EffV2S has passed EffB0 on classifier val top1. Wait for its eval and full pipeline grid "
                "before replacing the current completed pipeline best."
            )

    return {
        "generated_at": now(),
        "snapshot_dir": rel(SNAPSHOT_DIR),
        "detector": detector,
        "classifiers": classifiers,
        "pipeline_tops": pipeline_rows,
        "best_pipeline": best_pipeline,
        "active": active_process_summary(),
        "decision": decision,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--append-status", action="store_true")
    args = parser.parse_args()

    summary = build_summary()
    (SNAPSHOT_DIR / "current_best_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_readme(summary)
    if args.append_status:
        append_status_once(summary)

    best = summary.get("best_pipeline") or {}
    tf = summary["classifiers"].get("tf_effv2s_224_fixed", {})
    print(
        json.dumps(
            {
                "generated_at": summary["generated_at"],
                "snapshot_dir": summary["snapshot_dir"],
                "best_pipeline_e2e_f1": best.get("e2e_f1"),
                "best_pipeline_box_f1": best.get("box_f1"),
                "tf_epochs_completed": tf.get("epochs_completed"),
                "tf_best_val_top1": tf.get("best_val_top1"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
