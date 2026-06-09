#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import re
import subprocess


ROOT = Path("/home/admin/Sais_ocr")
E32_DIR = ROOT / "runs/detect/runs/detect/fixed_data_20260608/yolo11l_full_fixed_e32_safe_from_smoke"
RESUME_B1_DIR = ROOT / "runs/detect/runs/detect/fixed_data_20260608/yolo11l_full_fixed_e32_resume_b1_from_epoch9_best"
E1_DIR = ROOT / "runs/detect/runs/detect/fixed_data_20260608/yolo11l_full_fixed_e1_smoke_tmux"
E32_LOG = ROOT / "work/logs/yolo11l_full_fixed_e32_safe.log"
RESUME_B1_LOG = ROOT / "work/logs/yolo11l_full_fixed_e32_resume_b1.log"
EPOCH5_EVAL = ROOT / "work/evals/yolo11l_e32_epoch5_box_summary.json"
E32_EVAL = ROOT / "work/evals/yolo11l_e32_safe_box_summary.json"
RESUME_B1_EVAL = ROOT / "work/evals/yolo11l_e32_resume_b1_box_summary.json"
CLASSIFIER_CROP_DIR = ROOT / "work/classifier_crops_fixed_20260608"
CLASSIFIER_CROP_LOG = ROOT / "work/logs/prepare_classifier_crops_fixed_20260608.log"
PREFETCH_STATUS = ROOT / "work/logs/classifier_pretrained_prefetch_status.json"
CLASSIFIER_RUNS = [
    ROOT / "runs/classifier/effb0_128_fixed",
    ROOT / "runs/classifier/tf_effv2s_224_fixed",
    ROOT / "runs/classifier/smoke_effb0_64_cpu",
]


ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
PROGRESS_RE = re.compile(
    r"(?P<epoch>\d+)/(?P<epochs>\d+)\s+"
    r"(?P<gpu>[0-9.]+G)\s+"
    r"(?P<box>[0-9.]+)\s+"
    r"(?P<cls>[0-9.]+)\s+"
    r"(?P<dfl>[0-9.]+).*?"
    r"(?P<pct>\d+)%"
)
CROP_PROGRESS_RE = re.compile(
    r"prepare classifier (?P<split>train|val):\s+"
    r"(?P<pct>\d+)%.*?\|\s*"
    r"(?P<done>\d+)/(?P<total>\d+)"
)


def run(command: list[str]) -> str:
    try:
        return subprocess.check_output(command, cwd=ROOT, text=True, stderr=subprocess.STDOUT).strip()
    except subprocess.CalledProcessError as exc:
        return exc.output.strip()


def read_tail(path: Path, lines: int = 200) -> str:
    if not path.exists():
        return ""
    output = run(["tail", f"-{lines}", str(path)])
    return ANSI_RE.sub("", output)


def read_tail_bytes(path: Path, max_bytes: int = 200_000) -> str:
    if not path.exists():
        return ""
    with path.open("rb") as f:
        try:
            f.seek(-max_bytes, os.SEEK_END)
        except OSError:
            f.seek(0)
        data = f.read()
    text = data.decode("utf-8", errors="replace").replace("\r", "\n")
    return ANSI_RE.sub("", text)


def latest_progress(log_path: Path) -> dict | None:
    text = read_tail(log_path, 300)
    latest = None
    for line in text.splitlines():
        match = PROGRESS_RE.search(line)
        if not match:
            continue
        latest = {
            "epoch": int(match.group("epoch")),
            "epochs": int(match.group("epochs")),
            "gpu_mem": match.group("gpu"),
            "box_loss": float(match.group("box")),
            "cls_loss": float(match.group("cls")),
            "dfl_loss": float(match.group("dfl")),
            "epoch_progress_pct": int(match.group("pct")),
        }
    return latest


def latest_results(run_dir: Path) -> dict | None:
    results = run_dir / "results.csv"
    if not results.exists():
        return None
    with results.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[-1] if rows else None


def list_checkpoints(run_dir: Path) -> list[dict]:
    weights = run_dir / "weights"
    if not weights.exists():
        return []
    items = []
    for path in sorted(weights.glob("*.pt")):
        stat = path.stat()
        items.append({"name": path.name, "size_mb": round(stat.st_size / 1024 / 1024, 2)})
    return items


def best_eval(path: Path) -> dict | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list) and data:
        return data[0]
    if isinstance(data, dict) and "summary" in data:
        return data["summary"]
    return None


def read_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"error": str(exc), "path": str(path)}


def count_files(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for _, _, files in os.walk(path):
        total += len(files)
    return total


def classifier_crop_status() -> dict:
    status: dict = {
        "path": str(CLASSIFIER_CROP_DIR),
        "exists": CLASSIFIER_CROP_DIR.exists(),
        "stats_exists": (CLASSIFIER_CROP_DIR / "stats.json").exists(),
        "class_mapping_exists": (CLASSIFIER_CROP_DIR / "class_mapping.json").exists(),
        "train_files": count_files(CLASSIFIER_CROP_DIR / "train"),
        "val_files": count_files(CLASSIFIER_CROP_DIR / "val"),
    }
    stats_path = CLASSIFIER_CROP_DIR / "stats.json"
    if stats_path.exists():
        try:
            status["stats"] = json.loads(stats_path.read_text(encoding="utf-8"))
        except Exception as exc:
            status["stats_error"] = str(exc)

    text = read_tail_bytes(CLASSIFIER_CROP_LOG)
    latest = None
    for match in CROP_PROGRESS_RE.finditer(text):
        latest = {
            "split": match.group("split"),
            "percent": int(match.group("pct")),
            "done_images": int(match.group("done")),
            "total_images": int(match.group("total")),
        }
    status["latest_log_progress"] = latest
    return status


def classifier_run_status(path: Path) -> dict:
    status: dict = {
        "path": str(path),
        "exists": path.exists(),
        "best_exists": (path / "best.pth").exists(),
        "last_exists": (path / "last.pth").exists(),
        "metrics_exists": (path / "metrics.csv").exists(),
        "config_exists": (path / "config.json").exists(),
    }
    config = read_json(path / "config.json")
    if config is not None:
        status["config"] = config

    metrics_path = path / "metrics.csv"
    if metrics_path.exists():
        try:
            with metrics_path.open("r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
            status["metric_rows"] = len(rows)
            status["latest_metric"] = rows[-1] if rows else None
            status["best_val_top1"] = max(
                (float(row.get("val_top1", "nan")) for row in rows),
                default=None,
            )
        except Exception as exc:
            status["metrics_error"] = str(exc)
    return status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    status = {
        "tmux": run(["tmux", "list-sessions"]),
        "gpu": run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"]),
        "e1_results": latest_results(E1_DIR),
        "e32_progress": latest_progress(E32_LOG),
        "e32_results": latest_results(E32_DIR),
        "e32_checkpoints": list_checkpoints(E32_DIR),
        "resume_b1_progress": latest_progress(RESUME_B1_LOG),
        "resume_b1_results": latest_results(RESUME_B1_DIR),
        "resume_b1_checkpoints": list_checkpoints(RESUME_B1_DIR),
        "epoch5_best_eval": best_eval(EPOCH5_EVAL),
        "e32_best_eval": best_eval(E32_EVAL),
        "resume_b1_best_eval": best_eval(RESUME_B1_EVAL),
        "classifier_crops": classifier_crop_status(),
        "classifier_pretrained_prefetch": read_json(PREFETCH_STATUS),
        "classifier_runs": [classifier_run_status(path) for path in CLASSIFIER_RUNS],
    }

    if args.json:
        print(json.dumps(status, ensure_ascii=False, indent=2))
        return

    print("tmux:")
    print(status["tmux"] or "(none)")
    print("\ngpu:")
    print(status["gpu"] or "(none)")
    print("\ne1 last result:")
    print(json.dumps(status["e1_results"], ensure_ascii=False, indent=2))
    print("\ne32 progress:")
    print(json.dumps(status["e32_progress"], ensure_ascii=False, indent=2))
    print("\ne32 latest result:")
    print(json.dumps(status["e32_results"], ensure_ascii=False, indent=2))
    print("\ne32 checkpoints:")
    print(json.dumps(status["e32_checkpoints"], ensure_ascii=False, indent=2))
    print("\nresume_b1 progress:")
    print(json.dumps(status["resume_b1_progress"], ensure_ascii=False, indent=2))
    print("\nresume_b1 latest result:")
    print(json.dumps(status["resume_b1_results"], ensure_ascii=False, indent=2))
    print("\nresume_b1 checkpoints:")
    print(json.dumps(status["resume_b1_checkpoints"], ensure_ascii=False, indent=2))
    print("\nepoch5 best box eval:")
    print(json.dumps(status["epoch5_best_eval"], ensure_ascii=False, indent=2))
    print("\ne32 best box eval:")
    print(json.dumps(status["e32_best_eval"], ensure_ascii=False, indent=2))
    print("\nresume_b1 best box eval:")
    print(json.dumps(status["resume_b1_best_eval"], ensure_ascii=False, indent=2))
    print("\nclassifier crops:")
    print(json.dumps(status["classifier_crops"], ensure_ascii=False, indent=2))
    print("\nclassifier pretrained prefetch:")
    print(json.dumps(status["classifier_pretrained_prefetch"], ensure_ascii=False, indent=2))
    print("\nclassifier runs:")
    print(json.dumps(status["classifier_runs"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
