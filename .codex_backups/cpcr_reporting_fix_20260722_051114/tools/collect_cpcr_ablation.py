#!/usr/bin/env python3
"""Aggregate CPCR test summaries and evaluate the documented L4-to-L5 launch gate."""

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

from cpcr_experiments import ALL_STAGES, BASELINE_PER_SEED, K1_REFERENCE, L4_THRESHOLDS, STRUCTURES  # noqa: E402

METRICS = {"P": "metrics/precision(B)", "R": "metrics/recall(B)", "mAP50": "metrics/mAP50(B)", "mAP75": "metrics/mAP75(B)", "mAP50-95": "metrics/mAP50-95(B)"}
SCALE_METRICS = ("APS", "APM", "APL")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stages", nargs="+", choices=ALL_STAGES, required=True)
    parser.add_argument("--check-l4", action="store_true", help="Return 0 only when the L4 success gate permits L5.")
    return parser.parse_args()


def stats(values: list[float]) -> dict[str, float]:
    return {"mean": statistics.mean(values), "std": statistics.stdev(values) if len(values) > 1 else 0.0, "min": min(values), "max": max(values)}


def evaluate_l4(rows: list[dict]) -> dict:
    means = {name: statistics.mean(float(row[name]) for row in rows) for name in (*METRICS, *SCALE_METRICS)}
    checks = {f"mean_{name}": means[name] >= threshold for name, threshold in L4_THRESHOLDS.items()}
    checks["mAP50-95_std_le_0.25"] = statistics.stdev(float(row["mAP50-95"]) for row in rows) <= 0.25
    checks["gflops_le_6.70"] = float(rows[0]["GFLOPs"]) <= 6.70
    checks["mAP50-95_beats_baseline_2_of_3"] = sum(float(row["mAP50-95"]) > BASELINE_PER_SEED[int(row["seed"])]["mAP50-95"] for row in rows) >= 2
    checks["recall_not_below_baseline_2_of_3"] = sum(float(row["R"]) >= BASELINE_PER_SEED[int(row["seed"])]["R"] for row in rows) >= 2
    checks["mean_mAP50-95_beats_K1"] = means["mAP50-95"] > K1_REFERENCE["mean_mAP50-95"]
    checks["best_mAP50-95_beats_K1"] = max(float(row["mAP50-95"]) for row in rows) > K1_REFERENCE["best_mAP50-95"]
    return {"passed": all(checks.values()), "checks": checks, "means_percent": means, "reference": {"baseline_per_seed": BASELINE_PER_SEED, "k1": K1_REFERENCE}}


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
            records.append(row)
    output_dir = root / "runs" / "test"
    summary = {"run_id": args.run_id, "dataset": "/home/room305/ZZF/URPC2020half/data.yaml", "settings": {"epochs": 300, "patience": 40, "amp": False, "workers": 2, "plots": False, "pin_memory": False}, "stages": {}}
    for stage in args.stages:
        rows = [row for row in records if row["stage"] == stage]
        stage_summary = {"structure": STRUCTURES[stage], "n": len(rows), "runs": rows,
                         "metrics_percent": {label: stats([float(row[label]) for row in rows]) for label in (*METRICS, *SCALE_METRICS)},
                         "Params": rows[0]["Params"], "GFLOPs": rows[0]["GFLOPs"]}
        if stage == "l4":
            stage_summary["l5_launch_gate"] = evaluate_l4(rows)
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
    if args.check_l4:
        if "l4" not in args.stages:
            raise ValueError("--check-l4 requires l4 in --stages")
        raise SystemExit(0 if summary["stages"]["l4"]["l5_launch_gate"]["passed"] else 3)


if __name__ == "__main__":
    main()
