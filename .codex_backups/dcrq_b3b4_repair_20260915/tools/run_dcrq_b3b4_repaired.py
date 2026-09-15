#!/usr/bin/env python3
"""Fail-fast three-seed manager for the validated, same-index repaired DCRQ B3 and B4 reruns."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


PARENT_STAGE = "B2_D_RPQC_PARENT"
STAGES = ("B3_DAWRB_D_RPQC", "B4_DCRQ_FULL")
YAMLS = {"B3_DAWRB_D_RPQC": "yolov13n-dcrq-b3.yaml", "B4_DCRQ_FULL": "yolov13n-dcrq-b4.yaml"}
STRUCTURE = {
    PARENT_STAGE: {"source": "completed DCRQ B2_D_RPQC"},
    "B3_DAWRB_D_RPQC": {"layer4": "DAWRBDSC3k2(DSC3k2→DAWRB)", "layer20": "Concat", "head": "RPQUDQDetect"},
    "B4_DCRQ_FULL": {"layer4": "DAWRBDSC3k2(DSC3k2→DAWRB)", "layer20": "RCFConcat", "head": "RPQUDQDetect"},
}
SETTINGS = {
    "epochs": 160, "batch": 16, "imgsz": 640, "device": 0, "workers": 2,
    "amp": False, "deterministic": True, "plots": False, "patience": "not set",
    "parallel_seed_processes": 3, "seeds_per_stage": 3,
    "initialization": "YOLO(model_yaml).load(yolov13n.pt)",
}
METRICS = {
    "P": "metrics/precision(B)", "R": "metrics/recall(B)", "mAP50": "metrics/mAP50(B)",
    "mAP75": "metrics/mAP75(B)", "mAP50-95": "metrics/mAP50-95(B)",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def absolute(path: Path, label: str) -> Path:
    if not path.is_absolute():
        raise ValueError(f"{label} must be absolute")
    return path.resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--parent-run-root", type=Path, required=True)
    parser.add_argument("--coverage-report", type=Path, required=True)
    return parser.parse_args()


def train_complete(run_root: Path, stage: str, seed: int) -> bool:
    folder = run_root / "train" / stage / f"seed{seed}"
    return (folder / "weights" / "best.pt").is_file() and (folder / "train_complete.json").is_file()


def test_complete(run_root: Path, stage: str, seed: int) -> bool:
    folder = run_root / "test" / stage / f"seed{seed}"
    return (folder / "summary_metrics.json").is_file() and (folder / "scale_ap_metrics.json").is_file()


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


def archive_incomplete(run_root: Path, stage: str, seed: int) -> None:
    target = run_root / "train" / stage / f"seed{seed}"
    if not target.exists() or train_complete(run_root, stage, seed):
        return
    archive = run_root / "train" / "_failed_attempts" / stage / f"seed{seed}_{datetime.now().strftime('%Y%m%dT%H%M%SZ')}"
    archive.parent.mkdir(parents=True, exist_ok=True)
    target.replace(archive)


def load_metrics(run_root: Path, stage: str, seed: int) -> dict:
    summary = json.loads((run_root / "test" / stage / f"seed{seed}" / "summary_metrics.json").read_text(encoding="utf-8"))
    metrics, scale = summary["metrics"], summary["scale_metrics_percent"]
    row = {"seed": seed}
    for output, source in METRICS.items():
        row[output] = 100.0 * float(metrics[source])
    for name in ("APS", "APM", "APL"):
        row[name] = float(scale[name])
    model = summary.get("model", {})
    row["Params"], row["GFLOPs"] = int(model.get("parameters", 0)), float(model.get("gflops", 0.0))
    return row


def summarize(run_root: Path, stage: str) -> dict:
    rows = [load_metrics(run_root, stage, seed) for seed in (0, 1, 2)]
    names = [key for key in rows[0] if key != "seed"]
    mean = {key: sum(row[key] for row in rows) / 3 for key in names}
    summary = {
        "stage": stage, "structure": STRUCTURE[stage], "seeds": rows, "mean": mean,
        "std": {key: (sum((row[key] - mean[key]) ** 2 for row in rows) / 3) ** 0.5 for key in names},
        "best": {key: max(row[key] for row in rows) for key in names},
        "worst": {key: min(row[key] for row in rows) for key in names}, "generated_at": now(),
    }
    target = run_root / "test" / stage
    atomic_json(target / "all_seed_summary.json", summary)
    with (target / "all_seed_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return summary


def copy_parent_b2(run_root: Path, parent_root: Path) -> None:
    destination = run_root / "test" / PARENT_STAGE
    if (destination / "all_seed_summary.json").is_file():
        return
    source = parent_root / "test" / "B2_D_RPQC"
    for seed in (0, 1, 2):
        for name in ("summary_metrics.json", "scale_ap_metrics.json"):
            source_file = source / f"seed{seed}" / name
            if not source_file.is_file():
                raise FileNotFoundError(f"B2 parent artifact missing: {source_file}")
            destination_seed = destination / f"seed{seed}"
            destination_seed.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_file, destination_seed / name)
    summary = summarize(run_root, PARENT_STAGE)
    atomic_json(destination / "parent_provenance.json", {"source": str(source), "summary": summary, "copied_at": now()})


def paired_difference(run_root: Path, stage: str) -> None:
    parent = {row["seed"]: row for row in summarize(run_root, PARENT_STAGE)["seeds"]}
    child = {row["seed"]: row for row in summarize(run_root, stage)["seeds"]}
    metric_names = [key for key in child[0] if key not in {"seed", "Params", "GFLOPs"}]
    rows = [{"seed": seed, **{key: child[seed][key] - parent[seed][key] for key in metric_names}} for seed in (0, 1, 2)]
    mean = {key: sum(row[key] for row in rows) / 3 for key in metric_names}
    target = run_root / "test" / stage
    atomic_json(target / "paired_seed_difference_vs_B2.json", {"parent": PARENT_STAGE, "stage": stage, "seeds": rows, "mean": mean})
    with (target / "paired_seed_difference_vs_B2.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    root, data, parent, coverage = absolute(args.root, "root"), absolute(args.data, "data"), absolute(args.parent_run_root, "parent-run-root"), absolute(args.coverage_report, "coverage-report")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{2,80}", args.run_id):
        raise ValueError("invalid run-id")
    if not data.is_file() or not coverage.is_file() or not (root / "yolov13n.pt").is_file():
        raise FileNotFoundError("data, coverage report, or yolov13n.pt is missing")
    parent_state = json.loads((parent / "state.json").read_text(encoding="utf-8"))
    if parent_state.get("status") != "completed":
        raise RuntimeError("DCRQ B2 parent run must be completed")
    coverage_data = json.loads(coverage.read_text(encoding="utf-8"))
    if not all(coverage_data.get(stage, {}).get("equal_to_b2") for stage in STAGES):
        raise RuntimeError("validated B3/B4 coverage equality is absent")
    run_root = root / "runs" / f"dcrq_b3b4_repaired_urpc2019_{args.run_id}"
    state_path, current_stage, active_pids = run_root / "state.json", "", []
    launcher_pid = int(os.environ.get("DCRQ_B3B4_LAUNCHER_PID", os.getpid()))

    def write_state(status: str, detail: str = "", failure: str = "") -> None:
        atomic_json(state_path, {"run_id": args.run_id, "status": status, "stage": current_stage, "dataset": str(data),
            "parent_run_root": str(parent), "parent_status": "completed", "launcher_pid": launcher_pid, "worker_pids": active_pids,
            "completed_stages": [stage for stage in (PARENT_STAGE, *STAGES) if (run_root / "test" / stage / "all_seed_summary.json").is_file()],
            "settings": SETTINGS, "detail": detail, "failure_reason": failure, "updated_at": now()})

    def run_stage(stage: str) -> None:
        nonlocal current_stage, active_pids
        current_stage = stage
        if (run_root / "test" / stage / "all_seed_summary.json").is_file():
            return
        for seed in (0, 1, 2):
            archive_incomplete(run_root, stage, seed)
        processes = {}
        for seed in (0, 1, 2):
            if train_complete(run_root, stage, seed):
                continue
            log_path = run_root / "train" / stage / f"seed{seed}.worker.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handle = log_path.open("a", encoding="utf-8")
            command = [sys.executable, str(root / "tools" / "train_dcrq_worker.py"), "--root", str(root), "--run-root", str(run_root),
                       "--stage", stage, "--seed", str(seed), "--data", str(data)]
            processes[seed] = (subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, cwd=str(root), start_new_session=True), handle)
        active_pids = [process.pid for process, _ in processes.values()]
        write_state("running", f"Training {stage} with three concurrent seeds")
        failure = ""
        while processes:
            for seed, (process, handle) in list(processes.items()):
                code = process.poll()
                if code is None:
                    continue
                handle.close()
                del processes[seed]
                if code != 0:
                    failure = f"{stage} seed{seed} exited with code {code}"
                    break
            if failure:
                for process, handle in processes.values():
                    terminate_group(process)
                    handle.close()
                break
            time.sleep(5)
        active_pids = []
        if failure:
            raise RuntimeError(failure)
        for seed in (0, 1, 2):
            if not train_complete(run_root, stage, seed):
                raise RuntimeError(f"{stage} seed{seed} lacks required training artifacts")
        for seed in (0, 1, 2):
            if test_complete(run_root, stage, seed):
                continue
            log_path = run_root / "test" / stage / f"seed{seed}.worker.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, str(root / "test.py"), "--weights", str(run_root / "train" / stage / f"seed{seed}" / "weights" / "best.pt"),
                       "--data", str(data), "--name", f"seed{seed}", "--device", "0", "--batch", "16", "--imgsz", "640", "--workers", "2",
                       "--project", str(run_root / "test" / stage), "--no-plots"]
            with log_path.open("a", encoding="utf-8") as handle:
                code = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, cwd=str(root), check=False).returncode
            if code != 0 or not test_complete(run_root, stage, seed):
                raise RuntimeError(f"{stage} seed{seed} validation failed with code {code}")
        summarize(run_root, stage)
        paired_difference(run_root, stage)
        write_state("running", f"Completed {stage} training, testing, summary, and paired B2 differences")

    try:
        run_root.mkdir(parents=True, exist_ok=True)
        (run_root / "launcher.pid").write_text(f"{launcher_pid}\n", encoding="utf-8")
        atomic_json(run_root / "metadata.json", {"run_id": args.run_id, "dataset": str(data), "dataset_sha256": sha256(data), "parent": str(parent),
            "repair": "same-index DAWRBDSC3k2", "coverage_report": str(coverage), "coverage": coverage_data, "settings": SETTINGS,
            "structure": STRUCTURE, "created_at": now()})
        snapshot = run_root / "train" / "config_snapshots"
        snapshot.mkdir(parents=True, exist_ok=True)
        for yaml_name in YAMLS.values():
            shutil.copy2(root / "ultralytics" / "cfg" / "models" / "v13" / yaml_name, snapshot / yaml_name)
        write_state("running", "Validated repaired B3/B4 rerun initialized")
        copy_parent_b2(run_root, parent)
        for stage in STAGES:
            run_stage(stage)
        current_stage = ""
        write_state("completed", "Repaired B3 and B4 completed with three-seed summaries and paired B2 deltas")
    except BaseException as error:
        active_pids = []
        write_state("failed", str(error), str(error))
        raise


if __name__ == "__main__":
    main()
