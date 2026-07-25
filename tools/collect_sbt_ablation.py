#!/usr/bin/env python3
"""Aggregate completed SBT test summaries; T0 is represented by user-designated reused baselines."""

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

from tools.sbt_experiments import REFERENCE_SUMMARIES, STRUCTURES, TRAIN_STAGES  # noqa: E402

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
    parser.add_argument("--stages", nargs="+", choices=TRAIN_STAGES, required=True)
    return parser.parse_args()


def stats(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def read_references() -> dict[str, object]:
    result: dict[str, object] = {}
    for label, raw_path in REFERENCE_SUMMARIES.items():
        path = Path(raw_path)
        if not path.is_file():
            result[label] = {"path": str(path), "available": False}
            continue
        result[label] = {"path": str(path), "available": True, "summary": json.loads(path.read_text(encoding="utf-8"))}
    return result


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    records: list[dict[str, object]] = []
    for stage in args.stages:
        for seed in (0, 1, 2):
            name = f"{args.run_id}_{stage}_seed{seed}"
            path = root / "runs" / "test" / name / "summary_metrics.json"
            if not path.is_file():
                raise FileNotFoundError(path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            row: dict[str, object] = {
                "stage": stage,
                "seed": seed,
                "test_name": name,
                "structure": STRUCTURES[stage],
                "weights": payload["weights"],
                "Params": int(payload["model"].get("parameters", 0)),
                "GFLOPs": float(payload["model"].get("gflops", 0.0)),
            }
            row.update({label: float(payload["metrics"][key]) * 100.0 for label, key in METRICS.items()})
            row.update({label: float(payload.get("scale_metrics_percent", {}).get(label, 0.0)) for label in SCALE_METRICS})
            records.append(row)
    output_dir = root / "runs" / "test"
    summary: dict[str, object] = {
        "run_id": args.run_id,
        "dataset": "/home/room305/ZZF/URPC2020half/data.yaml",
        "settings": {"epochs": 300, "patience": 40, "amp": False, "workers": 2, "plots": False, "pin_memory": False},
        "t0_reused_baselines": read_references(),
        "stages": {},
    }
    for stage in args.stages:
        rows = [row for row in records if row["stage"] == stage]
        stage_summary = {
            "structure": STRUCTURES[stage],
            "n": len(rows),
            "runs": rows,
            "metrics_percent": {label: stats([float(row[label]) for row in rows]) for label in (*METRICS, *SCALE_METRICS)},
            "Params": rows[0]["Params"],
            "GFLOPs": rows[0]["GFLOPs"],
        }
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


if __name__ == "__main__":
    main()
