#!/usr/bin/env python3
"""Fail-fast BSRQ S1--S7 manager with verified reuse of the completed DCRQ B1 baseline."""

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


S0 = "S0_B1_UDQ_REUSED"
PHASE1 = ("S1_BSS", "S2_RAFULLPAD", "S3_SCQUDQ")
STAGE_YAMLS = {
    "S1_BSS": "yolov13n-bsrq-s1.yaml",
    "S2_RAFULLPAD": "yolov13n-bsrq-s2.yaml",
    "S3_SCQUDQ": "yolov13n-bsrq-s3.yaml",
    "S4_BSS_SCQUDQ": "yolov13n-bsrq-s4.yaml",
    "S5_RAFULLPAD_SCQUDQ": "yolov13n-bsrq-s5.yaml",
    "S6_BSS_RAFULLPAD": "yolov13n-bsrq-s6.yaml",
    "S7_BSRQ_FULL": "yolov13n-bsrq-s7.yaml",
}
STAGE_STRUCTURE = {
    S0: {"stem": "Conv", "p3_fullpad": "FullPAD_Tunnel", "detect": "UDQDetect", "source": "DCRQ B1_D"},
    "S1_BSS": {"stem": "BoundedSpectralStem", "p3_fullpad": "FullPAD_Tunnel", "detect": "UDQDetect"},
    "S2_RAFULLPAD": {"stem": "Conv", "p3_fullpad": "ReliabilityAdaptiveFullPAD", "detect": "UDQDetect"},
    "S3_SCQUDQ": {"stem": "Conv", "p3_fullpad": "FullPAD_Tunnel", "detect": "SCQUDQDetect"},
    "S4_BSS_SCQUDQ": {"stem": "BoundedSpectralStem", "p3_fullpad": "FullPAD_Tunnel", "detect": "SCQUDQDetect"},
    "S5_RAFULLPAD_SCQUDQ": {"stem": "Conv", "p3_fullpad": "ReliabilityAdaptiveFullPAD", "detect": "SCQUDQDetect"},
    "S6_BSS_RAFULLPAD": {"stem": "BoundedSpectralStem", "p3_fullpad": "ReliabilityAdaptiveFullPAD", "detect": "UDQDetect"},
    "S7_BSRQ_FULL": {"stem": "BoundedSpectralStem", "p3_fullpad": "ReliabilityAdaptiveFullPAD", "detect": "SCQUDQDetect"},
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
    "parallel_seed_processes": 3,
    "seeds_per_stage": 3,
    "initialization": "YOLO(model_yaml).load(yolov13n.pt)",
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


def require_absolute(path: Path, label: str) -> Path:
    if not path.is_absolute():
        raise ValueError(f"{label} must be absolute: {path}")
    return path.resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--parent-run-root", type=Path, required=True)
    parser.add_argument(
        "--force-all-phase2",
        action="store_true",
        help="Bypass the phase-2 go/no-go gate and run every S4--S7 combination after S1--S3.",
    )
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


def archive_incomplete_output(run_root: Path, stage: str, seed: int) -> None:
    output = run_root / "train" / stage / f"seed{seed}"
    if not output.exists() or train_complete(run_root, stage, seed):
        return
    archive = run_root / "train" / "_failed_attempts" / stage / f"seed{seed}_{datetime.now().strftime('%Y%m%dT%H%M%SZ')}"
    archive.parent.mkdir(parents=True, exist_ok=True)
    output.replace(archive)


def load_seed_metrics(run_root: Path, stage: str, seed: int) -> dict:
    summary_path = run_root / "test" / stage / f"seed{seed}" / "summary_metrics.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    metrics, scale = summary.get("metrics", {}), summary.get("scale_metrics_percent", {})
    row = {"seed": seed}
    for output_name, source_name in METRICS.items():
        row[output_name] = 100.0 * float(metrics[source_name])
    for name in ("APS", "APM", "APL"):
        row[name] = float(scale[name])
    model = summary.get("model", {})
    row["Params"] = int(model.get("parameters", 0))
    row["GFLOPs"] = float(model.get("gflops", 0.0))
    return row


def stage_summary(run_root: Path, stage: str) -> dict:
    rows = [load_seed_metrics(run_root, stage, seed) for seed in (0, 1, 2)]
    numeric = [key for key in rows[0] if key != "seed"]
    mean = {key: sum(row[key] for row in rows) / len(rows) for key in numeric}
    variance = {key: sum((row[key] - mean[key]) ** 2 for row in rows) / len(rows) for key in numeric}
    summary = {
        "stage": stage,
        "structure": STAGE_STRUCTURE[stage],
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
    return summary


def copy_parent_baseline(run_root: Path, parent_run_root: Path) -> None:
    destination = run_root / "test" / S0
    if (destination / "all_seed_summary.json").is_file():
        return
    source = parent_run_root / "test" / "B1_D"
    files = []
    for seed in (0, 1, 2):
        source_seed, destination_seed = source / f"seed{seed}", destination / f"seed{seed}"
        for name in ("summary_metrics.json", "scale_ap_metrics.json"):
            source_file = source_seed / name
            if not source_file.is_file():
                raise FileNotFoundError(f"Completed B1 baseline artifact missing: {source_file}")
            destination_seed.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_file, destination_seed / name)
            files.append({"source": str(source_file), "sha256": sha256(source_file), "destination": str(destination_seed / name)})
    summary = stage_summary(run_root, S0)
    atomic_json(
        destination / "baseline_provenance.json",
        {
            "reused_from": str(parent_run_root),
            "source_stage": "B1_D",
            "source_all_seed_summary": str(source / "all_seed_summary.json"),
            "copied_files": files,
            "verified_summary": summary,
            "reason": "S0 is the completed, protocol-matched DCRQ B1_D parent and is not retrained.",
        },
    )


def verify_completed_parent(parent_run_root: Path, data: Path) -> None:
    state_path = parent_run_root / "state.json"
    if not state_path.is_file():
        raise FileNotFoundError(f"DCRQ parent state is missing: {state_path}")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("status") != "completed":
        raise RuntimeError(f"DCRQ parent must be completed before reuse, got {state.get('status')!r}")
    for seed in (0, 1, 2):
        train_record = parent_run_root / "train" / "B1_D" / f"seed{seed}" / "train_complete.json"
        if not train_record.is_file():
            raise FileNotFoundError(f"Completed B1 train record missing: {train_record}")
        record = json.loads(train_record.read_text(encoding="utf-8"))
        settings = record.get("settings", {})
        expected = {key: SETTINGS[key] for key in ("epochs", "batch", "imgsz", "device", "workers", "amp", "deterministic", "plots")}
        if {key: settings.get(key) for key in expected} != expected or settings.get("patience") != "not set":
            raise RuntimeError(f"B1 seed{seed} protocol does not match the fixed BSRQ protocol")
        if Path(record.get("dataset", "")).resolve() != data:
            raise RuntimeError(f"B1 seed{seed} dataset differs from requested URPC2019 data: {record.get('dataset')}")


def decide_phase2(run_root: Path) -> tuple[list[str], dict]:
    baseline = json.loads((run_root / "test" / S0 / "all_seed_summary.json").read_text(encoding="utf-8"))
    s1 = json.loads((run_root / "test" / "S1_BSS" / "all_seed_summary.json").read_text(encoding="utf-8"))
    s2 = json.loads((run_root / "test" / "S2_RAFULLPAD" / "all_seed_summary.json").read_text(encoding="utf-8"))
    s3 = json.loads((run_root / "test" / "S3_SCQUDQ" / "all_seed_summary.json").read_text(encoding="utf-8"))
    base, mean1, mean2, mean3 = baseline["mean"], s1["mean"], s2["mean"], s3["mean"]
    bss_compensated = mean1["P"] >= base["P"] or mean1["R"] >= base["R"]
    bss_pass = not (mean1["mAP50-95"] < base["mAP50-95"] - 0.10 and not bss_compensated)
    ra_apl_not_clearly_down = mean2["APL"] >= base["APL"] - 0.10
    ra_pass = (mean2["APS"] > base["APS"] or s2["std"]["APS"] < baseline["std"]["APS"]) and ra_apl_not_clearly_down
    scq_pass = mean3["mAP75"] >= base["mAP75"] and mean3["mAP50-95"] >= base["mAP50-95"] and s3["std"]["APS"] < 1.12
    chosen = []
    if bss_pass and scq_pass:
        chosen.append("S4_BSS_SCQUDQ")
    if ra_pass and scq_pass:
        chosen.append("S5_RAFULLPAD_SCQUDQ")
    if bss_pass and ra_pass:
        chosen.append("S6_BSS_RAFULLPAD")
    if bss_pass and ra_pass and scq_pass:
        chosen.append("S7_BSRQ_FULL")
    decision = {
        "baseline": S0,
        "thresholds_percent_points": {
            "BSS_eliminate_if_mAP50_95_below": -0.10,
            "RA_APL_not_clearly_declined_below": -0.10,
            "SCQ_APS_std_must_be_below": 1.12,
        },
        "BSS": {"pass": bss_pass, "mAP50_95_delta": mean1["mAP50-95"] - base["mAP50-95"], "P_or_R_compensated": bss_compensated},
        "RAFullPAD": {"pass": ra_pass, "APS_delta": mean2["APS"] - base["APS"], "APS_std_delta": s2["std"]["APS"] - baseline["std"]["APS"], "APL_delta": mean2["APL"] - base["APL"]},
        "SCQ": {"pass": scq_pass, "mAP75_delta": mean3["mAP75"] - base["mAP75"], "mAP50_95_delta": mean3["mAP50-95"] - base["mAP50-95"], "APS_std": s3["std"]["APS"]},
        "selected_phase2_stages": chosen,
        "decided_at": utc_now(),
    }
    atomic_json(run_root / "phase2_gate_decision.json", decision)
    return chosen, decision


def main() -> None:
    args = parse_args()
    root, data, parent_run_root = (require_absolute(args.root, "root"), require_absolute(args.data, "data"), require_absolute(args.parent_run_root, "parent-run-root"))
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{2,80}", args.run_id):
        raise ValueError("run_id must contain only letters, numbers, underscores, and hyphens")
    if not data.is_file() or not parent_run_root.is_dir() or not (root / "yolov13n.pt").is_file():
        raise FileNotFoundError("Dataset, completed B1 parent, or yolov13n.pt is missing")
    for yaml_name in STAGE_YAMLS.values():
        if not (root / "ultralytics" / "cfg" / "models" / "v13" / yaml_name).is_file():
            raise FileNotFoundError(f"BSRQ model YAML missing: {yaml_name}")
    run_root = root / "runs" / f"bsrq_urpc2019_20260915_{args.run_id}"
    state_path = run_root / "state.json"
    launcher_pid = int(os.environ.get("BSRQ_LAUNCHER_PID", os.getpid()))
    current_stage, active_pids, phase2_decision = "", [], {}

    def write_state(status: str, detail: str = "", failed_reason: str = "") -> None:
        atomic_json(
            state_path,
            {
                "run_id": args.run_id,
                "status": status,
                "stage": current_stage,
                "dataset": str(data),
                "parent_run_root": str(parent_run_root),
                "parent_status": "completed (verified B1_D artifacts)",
                "launcher_pid": launcher_pid,
                "worker_pids": active_pids,
                "completed_stages": [stage for stage in (S0, *STAGE_YAMLS) if (run_root / "test" / stage / "all_seed_summary.json").is_file()],
                "settings": SETTINGS,
                "phase2_decision": phase2_decision,
                "detail": detail,
                "failure_reason": failed_reason,
                "updated_at": utc_now(),
            },
        )

    def run_stage(stage: str) -> None:
        nonlocal active_pids, current_stage
        current_stage = stage
        if (run_root / "test" / stage / "all_seed_summary.json").is_file():
            return
        for seed in (0, 1, 2):
            archive_incomplete_output(run_root, stage, seed)
        pending = [seed for seed in (0, 1, 2) if not train_complete(run_root, stage, seed)]
        processes: dict[int, tuple[subprocess.Popen, object]] = {}
        for seed in pending:
            log_path = run_root / "train" / stage / f"seed{seed}.worker.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handle = log_path.open("a", encoding="utf-8")
            command = [
                sys.executable,
                str(root / "tools" / "train_bsrq_worker.py"),
                "--root", str(root), "--run-root", str(run_root), "--stage", stage, "--seed", str(seed), "--data", str(data),
            ]
            processes[seed] = (subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, cwd=str(root), start_new_session=True), handle)
        active_pids = [process.pid for process, _ in processes.values()]
        write_state("running", f"Training {stage} with seeds {pending} in one concurrent three-process batch.")
        failure = ""
        while processes:
            for seed, (process, handle) in list(processes.items()):
                result = process.poll()
                if result is None:
                    continue
                handle.close()
                del processes[seed]
                if result != 0:
                    failure = f"{stage} seed{seed} exited with code {result}"
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
                raise RuntimeError(f"{stage} seed{seed} lacks best.pt or train_complete.json after worker exit")
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
                result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, cwd=str(root), check=False).returncode
            if result != 0 or not test_complete(run_root, stage, seed):
                raise RuntimeError(f"{stage} seed{seed} validation failed with code {result}")
        stage_summary(run_root, stage)
        write_state("running", f"Completed training, testing, and three-seed summary for {stage}")

    try:
        run_root.mkdir(parents=True, exist_ok=True)
        (run_root / "launcher.pid").write_text(f"{launcher_pid}\n", encoding="utf-8")
        verify_completed_parent(parent_run_root, data)
        atomic_json(
            run_root / "metadata.json",
            {
                "run_id": args.run_id,
                "dataset": str(data),
                "dataset_sha256": sha256(data),
                "parent_run_root": str(parent_run_root),
                "settings": SETTINGS,
                "stage_structure": STAGE_STRUCTURE,
                "created_at": utc_now(),
            },
        )
        snapshot_dir = run_root / "train" / "config_snapshots"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        for yaml_name in STAGE_YAMLS.values():
            shutil.copy2(root / "ultralytics" / "cfg" / "models" / "v13" / yaml_name, snapshot_dir / yaml_name)
        write_state("running", "BSRQ chain initialized; reusing the verified DCRQ B1 parent as S0")
        copy_parent_baseline(run_root, parent_run_root)
        for stage in PHASE1:
            run_stage(stage)
        if args.force_all_phase2:
            selected = ["S4_BSS_SCQUDQ", "S5_RAFULLPAD_SCQUDQ", "S6_BSS_RAFULLPAD", "S7_BSRQ_FULL"]
            phase2_decision = {
                "mode": "forced_all_phase2",
                "reason": "User explicitly cancelled BSRQ module gating.",
                "selected_phase2_stages": selected,
                "decided_at": utc_now(),
            }
            atomic_json(run_root / "phase2_gate_decision.json", phase2_decision)
        else:
            selected, phase2_decision = decide_phase2(run_root)
        write_state("running", f"Phase-2 gate selected {selected}")
        for stage in selected:
            run_stage(stage)
        current_stage = ""
        write_state("completed", "All selected BSRQ stages completed with three seed summaries")
    except BaseException as error:
        active_pids = []
        write_state("failed", str(error), str(error))
        raise


if __name__ == "__main__":
    main()
