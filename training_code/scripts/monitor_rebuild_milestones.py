#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path("/home/admin/Sais_ocr")
STATUS_MD = ROOT / "code/sais_ocr_rebuild_20260608/RUN_STATUS.md"
STATE_JSON = ROOT / "work/logs/rebuild_milestones_state.json"

DETECTOR_RUN = ROOT / (
    "runs/detect/runs/detect/fixed_data_20260608/"
    "yolo11l_full_fixed_e32_resume_b1_from_epoch9_best"
)
DETECTOR_RESULTS = DETECTOR_RUN / "results.csv"
DETECTOR_LOG = ROOT / "work/logs/yolo11l_full_fixed_e32_resume_b1.log"
ORIGINAL_E32_RESULTS = ROOT / (
    "runs/detect/runs/detect/fixed_data_20260608/"
    "yolo11l_full_fixed_e32_safe_from_smoke/results.csv"
)

SELECT_REPORT = ROOT / "work/evals/yolo11l_selected_detector_best.json"
SELECTED_WEIGHTS = ROOT / "work/snapshots/yolo11l_selected_detector_best.pt"
DETECTOR_BOX_SUMMARY = ROOT / "work/evals/yolo11l_selected_detector_box_summary.json"

CLASSIFIER_RUNS = {
    "effb0_128_fixed": ROOT / "runs/classifier/effb0_128_fixed",
    "tf_effv2s_224_fixed": ROOT / "runs/classifier/tf_effv2s_224_fixed",
}

ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
PROGRESS_RE = re.compile(
    r"(?P<epoch>\d+)/(?P<epochs>\d+)\s+"
    r"(?P<gpu>[0-9.]+G)\s+"
    r"(?P<box>[0-9.]+)\s+"
    r"(?P<cls>[0-9.]+)\s+"
    r"(?P<dfl>[0-9.]+).*?"
    r"(?P<pct>\d+)%"
)


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
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"error": str(exc), "path": str(path)}


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def metric(row: dict[str, str] | None, key: str) -> float | None:
    if row is None:
        return None
    try:
        value = float((row.get(key) or "nan").strip())
    except ValueError:
        return None
    if value != value:
        return None
    return value


def fmt(value: float | None) -> str:
    return "null" if value is None else f"{value:.5f}"


def latest_progress() -> dict[str, Any] | None:
    if not DETECTOR_LOG.exists():
        return None
    with DETECTOR_LOG.open("rb") as f:
        try:
            f.seek(-120_000, os.SEEK_END)
        except OSError:
            f.seek(0)
        text = f.read().decode("utf-8", errors="replace")
    latest = None
    for line in text.replace("\r", "\n").splitlines():
        clean = ANSI_RE.sub("", line)
        match = PROGRESS_RE.search(clean)
        if not match:
            continue
        latest = {
            "epoch": int(match.group("epoch")),
            "epochs": int(match.group("epochs")),
            "gpu": match.group("gpu"),
            "box": float(match.group("box")),
            "cls": float(match.group("cls")),
            "dfl": float(match.group("dfl")),
            "pct": int(match.group("pct")),
        }
    return latest


def tmux_sessions() -> str:
    try:
        return subprocess.check_output(
            ["tmux", "list-sessions"], cwd=ROOT, text=True, stderr=subprocess.STDOUT
        ).strip()
    except subprocess.CalledProcessError as exc:
        return exc.output.strip()


def gpu_processes() -> str:
    try:
        return subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            cwd=ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except subprocess.CalledProcessError as exc:
        return exc.output.strip()


def append_status(title: str, body: str) -> None:
    STATUS_MD.parent.mkdir(parents=True, exist_ok=True)
    with STATUS_MD.open("a", encoding="utf-8") as f:
        f.write(f"\n## {now()} {title}\n\n")
        f.write(body.rstrip() + "\n")


def detector_row_body(row: dict[str, str], prefix: str = "Completed detector epoch") -> str:
    return (
        f"{prefix}: `{row.get('epoch')}`.\n\n"
        f"- precision: `{fmt(metric(row, 'metrics/precision(B)'))}`\n"
        f"- recall: `{fmt(metric(row, 'metrics/recall(B)'))}`\n"
        f"- mAP50: `{fmt(metric(row, 'metrics/mAP50(B)'))}`\n"
        f"- mAP50-95: `{fmt(metric(row, 'metrics/mAP50-95(B)'))}`\n"
        f"- train box/cls/dfl: "
        f"`{fmt(metric(row, 'train/box_loss'))}` / "
        f"`{fmt(metric(row, 'train/cls_loss'))}` / "
        f"`{fmt(metric(row, 'train/dfl_loss'))}`\n"
    )


def best_detector_row(path: Path) -> dict[str, str] | None:
    rows = read_csv_rows(path)
    finite = [
        row
        for row in rows
        if metric(row, "metrics/mAP50-95(B)") is not None
        and metric(row, "metrics/mAP50(B)") is not None
    ]
    if not finite:
        return None
    return max(
        finite,
        key=lambda row: (
            metric(row, "metrics/mAP50-95(B)") or -1,
            metric(row, "metrics/mAP50(B)") or -1,
            metric(row, "metrics/recall(B)") or -1,
            metric(row, "metrics/precision(B)") or -1,
        ),
    )


def monitor_detector_results(state: dict[str, Any]) -> None:
    rows = read_csv_rows(DETECTOR_RESULTS)
    latest = rows[-1] if rows else None
    latest_epoch = latest.get("epoch") if latest else ""
    state_key = "detector_latest_epoch_logged"
    if latest and state.get(state_key) != latest_epoch:
        original_best = best_detector_row(ORIGINAL_E32_RESULTS)
        body = detector_row_body(latest)
        if original_best is not None:
            body += "\nReference original e32 best remains:\n\n"
            body += (
                f"- epoch: `{original_best.get('epoch')}`\n"
                f"- mAP50: `{fmt(metric(original_best, 'metrics/mAP50(B)'))}`\n"
                f"- mAP50-95: `{fmt(metric(original_best, 'metrics/mAP50-95(B)'))}`\n"
            )
        progress = latest_progress()
        if progress:
            body += "\nCurrent/last parsed progress line:\n\n"
            body += (
                f"- epoch: `{progress['epoch']}/{progress['epochs']}`\n"
                f"- progress: `{progress['pct']}%`\n"
                f"- loss box/cls/dfl: "
                f"`{progress['box']:.5f}` / `{progress['cls']:.5f}` / `{progress['dfl']:.5f}`\n"
            )
        append_status("detector epoch result", body)
        state[state_key] = latest_epoch

    if not state.get("detector_monitor_started"):
        progress = latest_progress()
        body = "Milestone monitor started. This monitor only reads logs/CSVs and does not use GPU.\n\n"
        body += f"- tmux sessions: `{tmux_sessions() or '(none)'}`\n"
        body += f"- gpu processes: `{gpu_processes() or '(none)'}`\n"
        if latest:
            body += "\nLatest completed detector row at monitor start:\n\n"
            body += detector_row_body(latest, prefix="Latest completed detector epoch")
        if progress:
            body += "\nCurrent parsed detector progress:\n\n"
            body += (
                f"- epoch: `{progress['epoch']}/{progress['epochs']}`\n"
                f"- progress: `{progress['pct']}%`\n"
                f"- loss box/cls/dfl: "
                f"`{progress['box']:.5f}` / `{progress['cls']:.5f}` / `{progress['dfl']:.5f}`\n"
            )
        append_status("milestone monitor active", body)
        state["detector_monitor_started"] = True


def file_signature(path: Path) -> str:
    if not path.exists():
        return ""
    stat = path.stat()
    return f"{stat.st_size}:{int(stat.st_mtime)}"


def monitor_selected_detector(state: dict[str, Any]) -> None:
    sig = file_signature(SELECT_REPORT)
    if not sig or state.get("select_report_sig") == sig:
        return
    report = read_json(SELECT_REPORT)
    selected = report.get("selected", {}) if isinstance(report, dict) else {}

    if "box_summary" in selected:
        box = selected.get("box_summary") or {}
        body = (
            "Selected detector checkpoint is available from multi-candidate box grid.\n\n"
            f"- output checkpoint: `{rel(SELECTED_WEIGHTS)}`\n"
            f"- selected candidate: `{selected.get('name')}`\n"
            f"- source checkpoint: `{selected.get('checkpoint')}`\n"
            f"- grid summary: `{report.get('summary')}`\n"
            f"- conf: `{box.get('conf')}`\n"
            f"- nms_iou: `{box.get('nms_iou')}`\n"
            f"- precision: `{fmt(box.get('precision'))}`\n"
            f"- recall: `{fmt(box.get('recall'))}`\n"
            f"- box f1: `{fmt(box.get('f1'))}`\n"
            f"- tp/fp/fn: `{box.get('tp')}` / `{box.get('fp')}` / `{box.get('fn')}`\n"
        )
    else:
        best = selected.get("best_row") or {}
        body = (
            "Selected detector checkpoint is available.\n\n"
            f"- output checkpoint: `{rel(SELECTED_WEIGHTS)}`\n"
            f"- selected candidate: `{selected.get('name')}`\n"
            f"- source checkpoint: `{selected.get('checkpoint')}`\n"
            f"- epoch: `{best.get('epoch')}`\n"
            f"- precision: `{fmt(metric(best, 'metrics/precision(B)'))}`\n"
            f"- recall: `{fmt(metric(best, 'metrics/recall(B)'))}`\n"
            f"- mAP50: `{fmt(metric(best, 'metrics/mAP50(B)'))}`\n"
            f"- mAP50-95: `{fmt(metric(best, 'metrics/mAP50-95(B)'))}`\n"
        )
    append_status("selected detector checkpoint", body)
    state["select_report_sig"] = sig


def monitor_detector_box_summary(state: dict[str, Any]) -> None:
    sig = file_signature(DETECTOR_BOX_SUMMARY)
    if not sig or state.get("detector_box_summary_sig") == sig:
        return
    data = read_json(DETECTOR_BOX_SUMMARY)
    top = data[0] if isinstance(data, list) and data else None
    if not isinstance(top, dict):
        return
    body = (
        "Selected detector box grid completed. Top row:\n\n"
        f"- summary: `{rel(DETECTOR_BOX_SUMMARY)}`\n"
        f"- candidate: `{top.get('candidate_name')}`\n"
        f"- checkpoint: `{top.get('candidate_checkpoint')}`\n"
        f"- conf: `{top.get('conf')}`\n"
        f"- nms_iou: `{top.get('nms_iou')}`\n"
        f"- precision: `{fmt(top.get('precision'))}`\n"
        f"- recall: `{fmt(top.get('recall'))}`\n"
        f"- f1: `{fmt(top.get('f1'))}`\n"
        f"- tp/fp/fn: `{top.get('tp')}` / `{top.get('fp')}` / `{top.get('fn')}`\n"
    )
    append_status("detector box grid result", body)
    state["detector_box_summary_sig"] = sig


def monitor_classifier_runs(state: dict[str, Any]) -> None:
    classifier_state = state.setdefault("classifier_runs", {})
    for name, run_dir in CLASSIFIER_RUNS.items():
        metrics_path = run_dir / "metrics.csv"
        rows = read_csv_rows(metrics_path)
        if not rows:
            continue
        latest = rows[-1]
        key = f"{name}:rows"
        if classifier_state.get(key) == len(rows):
            continue
        best_top1 = max((metric(row, "val_top1") or -1 for row in rows), default=-1)
        body = (
            f"Classifier `{name}` has a new completed epoch.\n\n"
            f"- metrics: `{rel(metrics_path)}`\n"
            f"- epoch: `{latest.get('epoch')}`\n"
            f"- train_loss: `{latest.get('train_loss')}`\n"
            f"- train_top1/top5: `{latest.get('train_top1')}` / `{latest.get('train_top5')}`\n"
            f"- val_loss: `{latest.get('val_loss')}`\n"
            f"- val_top1/top5: `{latest.get('val_top1')}` / `{latest.get('val_top5')}`\n"
            f"- best_val_top1_so_far: `{best_top1:.5f}`\n"
        )
        append_status("classifier epoch result", body)
        classifier_state[key] = len(rows)


def monitor_pipeline_summaries(state: dict[str, Any]) -> None:
    pipeline_state = state.setdefault("pipeline_summaries", {})
    for path in sorted((ROOT / "work/evals").glob("pipeline_*_summary.json")):
        sig = file_signature(path)
        if not sig or pipeline_state.get(str(path)) == sig:
            continue
        data = read_json(path)
        top = data[0] if isinstance(data, list) and data else None
        if not isinstance(top, dict):
            continue
        body = (
            "Pipeline grid summary updated. Top row:\n\n"
            f"- summary: `{rel(path)}`\n"
            f"- e2e_f1: `{fmt(top.get('e2e_f1'))}`\n"
            f"- e2e_precision: `{fmt(top.get('e2e_precision'))}`\n"
            f"- e2e_recall: `{fmt(top.get('e2e_recall'))}`\n"
            f"- box_f1: `{fmt(top.get('box_f1'))}`\n"
            f"- text_acc: `{fmt(top.get('text_accuracy_on_box_match'))}`\n"
            f"- det_conf/det_iou/pad: `{top.get('det_conf')}` / `{top.get('det_iou')}` / `{top.get('crop_padding')}`\n"
            f"- e2e tp/fp/fn: `{top.get('e2e_tp')}` / `{top.get('e2e_fp')}` / `{top.get('e2e_fn')}`\n"
        )
        append_status("pipeline grid result", body)
        pipeline_state[str(path)] = sig


def monitor_queue_complete(state: dict[str, Any]) -> None:
    log = ROOT / "work/logs/classifier_queue_fixed.log"
    if state.get("classifier_queue_complete") or not log.exists():
        return
    text = log.read_text(encoding="utf-8", errors="replace")[-20_000:]
    if "classifier queue complete" not in text:
        return
    append_status(
        "classifier queue complete",
        "Classifier queue reported completion.\n\n"
        f"- log: `{rel(log)}`\n"
        "- next step: compare pipeline summaries and build a candidate only if the gate is met.",
    )
    state["classifier_queue_complete"] = True


def process_once() -> None:
    state = read_json(STATE_JSON)
    if not isinstance(state, dict):
        state = {}

    monitor_detector_results(state)
    monitor_selected_detector(state)
    monitor_detector_box_summary(state)
    monitor_classifier_runs(state)
    monitor_pipeline_summaries(state)
    monitor_queue_complete(state)

    write_json(STATE_JSON, state)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=int, default=300)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    if args.once:
        process_once()
        return

    while True:
        try:
            process_once()
        except Exception as exc:
            append_status("milestone monitor error", f"`{type(exc).__name__}: {exc}`")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
