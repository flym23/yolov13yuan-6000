#!/usr/bin/env python3
"""Fail-fast, resumable three-seed DCRQ B0→B4 training and validation manager."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


STAGES = ("B0_L0", "B1_D", "B2_D_RPQC", "B3_DAWRB_D_RPQC", "B4_DCRQ_FULL")
YAMLS = {
    "B0_L0": "yolov13n-dcrq-b0.yaml",
    "B1_D": "yolov13n-dcrq-b1.yaml",
    "B2_D_RPQC": "yolov13n-dcrq-b2.yaml",
    "B3_DAWRB_D_RPQC": "yolov13n-dcrq-b3.yaml",
    "B4_DCRQ_FULL": "yolov13n-dcrq-b4.yaml",
}
SETTINGS = {
    "epochs": 160,
    "batch": 16,
    "imgsz": 640,
    "device": 0,
    "workers": 2,
    "amp": False,
    "deterministic": True,
    "plots": False,
    "patience": "not set",
    "parallel_seed_processes": 2,
    "seeds_per_stage": 3,
}
METRICS = {
    "P": "metrics/precision(B)",
    "R": "metrics/recall(B)",
    "mAP50": "metrics/mAP50(B)",
    "mAP75": "metrics/mAP75(B)",
    "mAP50-95": "metrics/mAP50-95(B)",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--data", type=Path, required=True)
    return parser.parse_args()


def require_absolute(path: Path, label: str) -> Path:
    if not path.is_absolute():
        raise ValueError(f"{label} must be absolute: {path}")
    return path.resolve()


def train_complete(run_root: Path, stage: str, seed: int) -> bool:
    path = run_root / "train" / stage / f"seed{seed}"
    return (path / "weights" / "best.pt").is_file() and (path / "train_complete.json").is_file()


def test_complete(run_root: Path, stage: str, seed: int) -> bool:
    path = run_root / "test" / stage / f"seed{seed}"
    return (path / "summary_metrics.json").is_file() and (path / "scale_ap_metrics.json").is_file()


def terminate_group(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=15)


def archive_incomplete_output(run_root: Path, stage: str, seed: int) -> None:
    """Preserve an interrupted seed before a clean retry; never delete experiments."""
    output = run_root / "train" / stage / f"seed{seed}"
    if not output.exists() or train_complete(run_root, stage, seed):
        return
    archive = run_root / "train" / "_failed_attempts" / stage / f"seed{seed}_{datetime.now().strftime('%Y%m%dT%H%M%SZ')}"
    archive.parent.mkdir(parents=True, exist_ok=True)
    output.replace(archive)


def load_seed_metrics(run_root: Path, stage: str, seed: int) -> dict:
    summary_path = run_root / "test" / stage / f"seed{seed}" / "summary_metrics.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    metrics = summary.get("metrics", {})
    scale = summary.get("scale_metrics_percent", {})
    row = {"seed": seed}
    for output_name, source_name in METRICS.items():
        row[output_name] = 100.0 * float(metrics[source_name])
    for name in ("APS", "APM", "APL"):
        row[name] = float(scale[name])
    model = summary.get("model", {})
    row["Params"] = int(model.get("parameters", 0))
    row["GFLOPs"] = float(model.get("gflops", 0.0))
    return row


def stage_summary(run_root: Path, stage: str) -> None:
    rows = [load_seed_metrics(run_root, stage, seed) for seed in (0, 1, 2)]
    numeric = [key for key in rows[0] if key != "seed"]
    mean = {key: sum(row[key] for row in rows) / len(rows) for key in numeric}
    variance = {key: sum((row[key] - mean[key]) ** 2 for row in rows) / len(rows) for key in numeric}
    summary = {
        "stage": stage,
        "seeds": rows,
        "mean": mean,
        "std": {key: variance[key] ** 0.5 for key in numeric},
        "best": {key: max(row[key] for row in rows) for key in numeric},
        "worst": {key: min(row[key] for row in rows) for key in numeric},
        "generated_at": utc_now(),
    }
    destination = run_root / "test" / stage
    atomic_json(destination / "all_seed_summary.json", summary)
    with (destination / "all_seed_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    root, data = require_absolute(args.root, "root"), require_absolute(args.data, "data")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{2,80}", args.run_id):
        raise ValueError("run_id must contain only letters, numbers, underscores, and hyphens.")
    if not data.is_file():
        raise FileNotFoundError(f"Dataset YAML missing: {data}")
    pretrained = root / "yolov13n.pt"
    if not pretrained.is_file():
        raise FileNotFoundError(f"Pretrained checkpoint missing: {pretrained}")
    for yaml_name in YAMLS.values():
        if not (root / "ultralytics" / "cfg" / "models" / "v13" / yaml_name).is_file():
            raise FileNotFoundError(f"DCRQ model YAML missing: {yaml_name}")

    run_root = root / "runs" / f"dcrq_urpc2019_20260914_{args.run_id}"
    state_path = run_root / "state.json"
    launcher_pid = int(os.environ.get("DCRQ_LAUNCHER_PID", os.getpid()))
    current_stage, active_pids = "", []

    def write_state(status: str, detail: str = "", failed_reason: str = "") -> None:
        atomic_json(
            state_path,
            {
                "run_id": args.run_id,
                "status": status,
                "stage": current_stage,
                "dataset": str(data),
                "upstream_state_path": None,
                "upstream_status": None,
                "upstream_failure_reason": None,
                "launcher_pid": launcher_pid,
                "worker_pids": active_pids,
                "completed_stages": [stage for stage in STAGES if (run_root / "test" / stage / "all_seed_summary.json").is_file()],
                "settings": SETTINGS,
                "detail": detail,
                "failure_reason": failed_reason,
                "updated_at": utc_now(),
            },
        )

    try:
        run_root.mkdir(parents=True, exist_ok=True)
        (run_root / "launcher.pid").write_text(f"{launcher_pid}\n", encoding="utf-8")
        write_state("running", "DCRQ ablation chain started or resumed.")
        for stage in STAGES:
            current_stage = stage
            if not (run_root / "test" / stage / "all_seed_summary.json").is_file():
                for seed in (0, 1, 2):
                    if train_complete(run_root, stage, seed):
                        continue
                    archive_incomplete_output(run_root, stage, seed)
                pending_seeds = [seed for seed in (0, 1, 2) if not train_complete(run_root, stage, seed)]
                for offset in range(0, len(pending_seeds), SETTINGS["parallel_seed_processes"]):
                    batch_seeds = pending_seeds[offset : offset + SETTINGS["parallel_seed_processes"]]
                    processes = {}
                    for seed in batch_seeds:
                        log_path = run_root / "train" / stage / f"seed{seed}.worker.log"
                        log_path.parent.mkdir(parents=True, exist_ok=True)
                        handle = log_path.open("a", encoding="utf-8")
                        command = [
                            sys.executable,
                            str(root / "tools" / "train_dcrq_worker.py"),
                            "--root", str(root), "--run-root", str(run_root), "--stage", stage, "--seed", str(seed), "--data", str(data),
                        ]
                        processes[seed] = (
                            subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True), handle
                        )
                    active_pids = [process.pid for process, _ in processes.values()]
                    write_state("running", f"Training {stage} with seeds {batch_seeds}; max parallel seeds=2.")
                    failed = None
                    while processes:
                        for seed, (process, handle) in list(processes.items()):
                            result = process.poll()
                            if result is None:
                                continue
                            handle.close()
                            del processes[seed]
                            if result != 0:
                                failed = f"{stage} seed{seed} exited with code {result}."
                                break
                        if failed:
                            for process, handle in processes.values():
                                terminate_group(process)
                                handle.close()
                            break
                        time.sleep(5)
                    active_pids = []
                    if failed:
                        write_state("failed", failed, failed)
                        raise RuntimeError(failed)
                for seed in (0, 1, 2):
                    if not train_complete(run_root, stage, seed):
                        raise RuntimeError(f"{stage} seed{seed} lacks best.pt or train_complete.json after worker exit.")

                for seed in (0, 1, 2):
                    if test_complete(run_root, stage, seed):
                        continue
                    log_path = run_root / "test" / stage / f"seed{seed}.worker.log"
                    log_path.parent.mkdir(parents=True, exist_ok=True)
                    command = [
                        sys.executable, str(root / "test.py"), "--weights", str(run_root / "train" / stage / f"seed{seed}" / "weights" / "best.pt"),
                        "--data", str(data), "--name", f"seed{seed}", "--device", "0", "--batch", "16", "--imgsz", "640", "--workers", "2",
                        "--project", str(run_root / "test" / stage), "--no-plots",
                    ]
                    with log_path.open("a", encoding="utf-8") as handle:
                        result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=False).returncode
                    if result != 0 or not test_complete(run_root, stage, seed):
                        failure = f"{stage} seed{seed} validation failed with code {result}."
                        write_state("failed", failure, failure)
                        raise RuntimeError(failure)
                stage_summary(run_root, stage)
                write_state("running", f"Completed training, testing, and all-seed summary for {stage}.")
        current_stage = ""
        write_state("completed", "All B0→B4 stages completed with three seed summaries.")
    except BaseException as error:
        active_pids = []
        write_state("failed", str(error), str(error))
        raise


if __name__ == "__main__":
    main()
