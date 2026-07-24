#!/usr/bin/env python3
"""Aggregate CBER test summaries and enforce the documented N4 launch gate."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tools.cber_experiments import H3_MEAN, H3_PER_SEED, L0_PER_SEED, N4_MAX_GFLOPS, N4_MAX_STD, N4_MINIMUMS, STAGE_ORDER, STRUCTURES  # noqa: E402

METRICS = {
    "P": "metrics/precision(B)",
    "R": "metrics/recall(B)",
    "mAP50": "metrics/mAP50(B)",
    "mAP75": "metrics/mAP75(B)",
    "mAP50-95": "metrics/mAP50-95(B)",
}
SCALE_METRICS = ("APS", "APM", "APL")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stages", nargs="+", choices=STAGE_ORDER, required=True)
    parser.add_argument("--check-n4", action="store_true", help="Exit 0 only when N4 qualifies N5/N6 training.")
    return parser.parse_args()


def mean_std(values: list[float]) -> dict[str, float]:
    return {"mean": statistics.mean(values), "std": statistics.stdev(values) if len(values) > 1 else 0.0, "min": min(values), "max": max(values)}


def diagnostic_summary(path: Path) -> dict[str, float | int]:
    if not path.is_file():
        return {"records": 0}
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        return {"records": 0}
    return {
        "records": len(records),
        "channel_gate_mean": statistics.mean(float(row["channel_gate"]["mean"]) for row in records),
        "channel_gate_max": max(float(row["channel_gate"]["max"]) for row in records),
        "spatial_gate_mean": statistics.mean(float(row["spatial_gate"]["mean"]) for row in records),
        "alpha_raw_last": float(records[-1]["alpha_raw"]),
    }


def evaluate_n4(rows: list[dict]) -> dict:
    means = {metric: statistics.mean(float(row[metric]) for row in rows) for metric in (*METRICS, *SCALE_METRICS)}
    checks = {f"mean_{metric}_minimum": means[metric] >= threshold for metric, threshold in N4_MINIMUMS.items()}
    checks["mAP50-95_std_le_0.15"] = statistics.stdev(float(row["mAP50-95"]) for row in rows) <= N4_MAX_STD
    checks["gflops_le_7.40"] = float(rows[0]["GFLOPs"]) <= N4_MAX_GFLOPS
    checks["mAP50-95_beats_l0_3_of_3"] = all(float(row["mAP50-95"]) > L0_PER_SEED[int(row["seed"])] for row in rows)
    checks["mAP50-95_beats_h3_2_of_3"] = sum(float(row["mAP50-95"]) > H3_PER_SEED[int(row["seed"])] for row in rows) >= 2
    checks["mean_mAP50-95_beats_h3"] = means["mAP50-95"] > H3_MEAN["mAP50-95"]
    checks["mean_APS_not_below_h3_minus_0.10"] = means["APS"] >= H3_MEAN["APS"] - 0.10
    return {"passed": all(checks.values()), "checks": checks, "means_percent": means, "references": {"l0_per_seed": L0_PER_SEED, "h3_per_seed": H3_PER_SEED, "h3_mean": H3_MEAN}}


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    records = []
    for stage in args.stages:
        for seed in (0, 1, 2):
            name = f"{args.run_id}_{stage}_seed{seed}"
            path = root / "runs" / "test" / name / "summary_metrics.json"
            if not path.is_file():
                raise FileNotFoundError(path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            row = {"stage": stage, "seed": seed, "test_name": name, "structure": STRUCTURES[stage], "weights": payload["weights"]}
            row.update({label: float(payload["metrics"][key]) * 100.0 for label, key in METRICS.items()})
            row.update({label: float(payload.get("scale_metrics_percent", {}).get(label, 0.0)) for label in SCALE_METRICS})
            row["Params"] = int(payload["model"].get("parameters", 0))
            row["GFLOPs"] = float(payload["model"].get("gflops", 0.0))
            row["diagnostics"] = diagnostic_summary(root / "runs" / "train" / f"{name}.cber_diagnostics.jsonl")
            records.append(row)
    output_dir = root / "runs" / "test"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "run_id": args.run_id,
        "dataset": "/home/room305/ZZF/URPC2020half/data.yaml",
        "settings": {"epochs": 300, "patience": 40, "amp": False, "workers": 2, "plots": False, "pin_memory": False},
        "stages": {},
    }
    for stage in args.stages:
        rows = [row for row in records if row["stage"] == stage]
        stage_summary = {
            "structure": STRUCTURES[stage],
            "n": len(rows),
            "runs": rows,
            "metrics_percent": {metric: mean_std([float(row[metric]) for row in rows]) for metric in (*METRICS, *SCALE_METRICS)},
            "Params": rows[0]["Params"],
            "GFLOPs": rows[0]["GFLOPs"],
        }
        if stage == "n4":
            stage_summary["n5_n6_launch_gate"] = evaluate_n4(rows)
        summary["stages"][stage] = stage_summary
        (output_dir / f"{args.run_id}_{stage}_summary.json").write_text(json.dumps(stage_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    fields = ("stage", "seed", "P", "R", "mAP50", "mAP75", "mAP50-95", "APS", "APM", "APL", "Params", "GFLOPs")
    csv_path = output_dir / f"{args.run_id}_ablation.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in records)
    summary["csv"] = str(csv_path)
    overview = output_dir / f"{args.run_id}_summary.json"
    overview.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(overview)
    if args.check_n4:
        if "n4" not in args.stages:
            raise ValueError("--check-n4 requires n4 in --stages")
        raise SystemExit(0 if summary["stages"]["n4"]["n5_n6_launch_gate"]["passed"] else 3)


if __name__ == "__main__":
    main()
