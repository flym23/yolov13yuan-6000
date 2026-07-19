#!/usr/bin/env python3
"""Aggregate reviewed URPC2020 baseline/DGDR test summaries for seed-wise analysis."""

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

from dgdr_experiments import STAGE_ORDER, STRUCTURES  # noqa: E402


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
    return parser.parse_args()


def _percent(value: float) -> float:
    return float(value) * 100.0


def _mean_std(values: list[float]) -> dict[str, float]:
    return {"mean": statistics.mean(values), "std": statistics.stdev(values) if len(values) > 1 else 0.0}


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    records = []
    for stage in args.stages:
        for seed in range(3):
            path = root / "runs" / "test" / f"{args.run_id}_{stage}_seed{seed}" / "summary_metrics.json"
            if not path.is_file():
                raise FileNotFoundError(path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            metrics = payload["metrics"]
            row = {"stage": stage, "seed": seed, "structure": STRUCTURES[stage], "weights": payload["weights"]}
            for name, key in METRICS.items():
                if key not in metrics:
                    raise KeyError(f"{path}: missing required metric {key}")
                row[name] = _percent(metrics[key])
            for name in SCALE_METRICS:
                row[name] = float(payload.get("scale_metrics_percent", {}).get(name, 0.0))
            model = payload.get("model", {})
            row["Params"] = int(model.get("parameters", 0))
            row["GFLOPs"] = float(model.get("gflops", 0.0))
            records.append(row)

    summary = {
        "run_id": args.run_id,
        "dataset": "/home/room305/ZZF/URPC2020half/data.yaml",
        "settings": {"epochs": 300, "patience": 40, "amp": False, "workers": 2, "plots": False},
        "stages": {},
    }
    for stage in args.stages:
        stage_rows = [row for row in records if row["stage"] == stage]
        summary["stages"][stage] = {
            "structure": STRUCTURES[stage],
            "n": len(stage_rows),
            "runs": stage_rows,
            "metrics_percent": {
                name: _mean_std([float(row[name]) for row in stage_rows])
                for name in (*METRICS, *SCALE_METRICS)
            },
            "Params": stage_rows[0]["Params"],
            "GFLOPs": stage_rows[0]["GFLOPs"],
        }
        path = root / "runs" / "test" / f"{args.run_id}_{stage}_summary.json"
        path.write_text(json.dumps(summary["stages"][stage], ensure_ascii=False, indent=2), encoding="utf-8")

    output_dir = root / "runs" / "dgdr_ablation"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{args.run_id}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    fields = ["stage", "seed", "P", "R", "mAP50", "mAP75", "mAP50-95", "APS", "APM", "APL", "Params", "GFLOPs"]
    with (output_dir / f"{args.run_id}_summary.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in records)
    print(output_dir / f"{args.run_id}_summary.json")


if __name__ == "__main__":
    main()
