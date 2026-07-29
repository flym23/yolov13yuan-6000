#!/usr/bin/env python3
"""Aggregate CMRF test summaries and preserve the two user-authorized historical references."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

from train_cmrf_worker import STAGES


METRICS = {
    "P": "metrics/precision(B)", "R": "metrics/recall(B)", "mAP50": "metrics/mAP50(B)",
    "mAP75": "metrics/mAP75(B)", "mAP50-95": "metrics/mAP50-95(B)",
}
SCALE_METRICS = ("APS", "APM", "APL")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stages", nargs="+", choices=tuple(STAGES), required=True)
    parser.add_argument("--l0-reference", type=Path, required=True)
    parser.add_argument("--p0-reference", type=Path, required=True)
    return parser.parse_args()


def mean_std(values: list[float]) -> dict[str, float]:
    return {"mean": statistics.mean(values), "std": statistics.stdev(values) if len(values) > 1 else 0.0}


def load_reference(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    rows = []
    for stage in args.stages:
        _, structure = STAGES[stage]
        for seed in (0, 1, 2):
            name = f"{args.run_id}_{stage}_seed{seed}"
            path = root / "runs" / "test" / name / "summary_metrics.json"
            if not path.is_file():
                raise FileNotFoundError(path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            row = {"stage": stage, "seed": seed, "test_name": name, "structure": structure, "weights": payload["weights"]}
            row.update({label: float(payload["metrics"][key]) * 100.0 for label, key in METRICS.items()})
            row.update({label: float(payload.get("scale_metrics_percent", {}).get(label, 0.0)) for label in SCALE_METRICS})
            row["Params"] = int(payload["model"].get("parameters", 0))
            row["GFLOPs"] = float(payload["model"].get("gflops", 0.0))
            rows.append(row)

    stages = {}
    for stage in args.stages:
        stage_rows = [row for row in rows if row["stage"] == stage]
        stages[stage] = {
            "structure": STAGES[stage][1], "n": len(stage_rows), "runs": stage_rows,
            "metrics_percent": {label: mean_std([float(row[label]) for row in stage_rows]) for label in (*METRICS, *SCALE_METRICS)},
            "Params": stage_rows[0]["Params"], "GFLOPs": stage_rows[0]["GFLOPs"],
        }
    output_dir = root / "runs" / "test"
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = ("stage", "seed", "P", "R", "mAP50", "mAP75", "mAP50-95", "APS", "APM", "APL", "Params", "GFLOPs")
    csv_path = output_dir / f"{args.run_id}_ablation.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in rows)
    result = {
        "run_id": args.run_id, "dataset": "/home/room305/ZZF/URPC2020half/data.yaml",
        "settings": {"epochs": 300, "patience": 40, "batch": 16, "imgsz": 640, "device": 0, "workers": 2,
                     "amp": False, "deterministic": True, "plots": False, "parallel_seed_processes": 3},
        "historical_references": {
            "l0": {"path": str(args.l0_reference), "summary": load_reference(args.l0_reference)},
            "p0": {"path": str(args.p0_reference), "summary": load_reference(args.p0_reference)},
            "note": "These supplied historical L0/P0 summaries are external anchors and are not relabelled as SCPG-H3 C0.",
        },
        "stages": stages, "csv": str(csv_path),
    }
    overview = output_dir / f"{args.run_id}_summary.json"
    overview.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(overview)


if __name__ == "__main__":
    main()
