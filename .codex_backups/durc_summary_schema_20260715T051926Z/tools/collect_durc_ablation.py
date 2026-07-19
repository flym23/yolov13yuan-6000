#!/usr/bin/env python3
"""Collect reviewed DURC metrics into an atomic JSON/CSV ablation summary."""

import argparse
import csv
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

from durc_experiments import MODEL_FILES, STAGE_ORDER


METRIC_KEYS = {
    "P": "metrics/precision(B)",
    "R": "metrics/recall(B)",
    "mAP50": "metrics/mAP50(B)",
    "mAP75": "metrics/mAP75(B)",
    "mAP50-95": "metrics/mAP50-95(B)",
}
SCALE_KEYS = ("APS", "APM", "APL")
RESULT_KEYS = (*METRIC_KEYS, *SCALE_KEYS, "Params", "GFLOPs")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--stages", nargs="+", choices=STAGE_ORDER, default=list(STAGE_ORDER)
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    return parser.parse_args()


def read_json(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def baseline_rows(root, baseline_complexity):
    candidates = (
        root / "runs/test/yolov13_baseline_20260714/yolov13_baseline_20260714.json",
        root / "yolov13_baseline_20260714.json",
    )
    baseline_path = next((path for path in candidates if path.is_file()), None)
    if baseline_path is None:
        return [], None
    payload = read_json(baseline_path)
    rows = []
    for item in payload.get("runs", []):
        row = {
            "stage": "A0",
            "repeat": int(item["run"]),
            "seed": None,
            "config": "yolov13.yaml",
            "test_output": item["test_output"],
            "P": float(item["precision"]),
            "R": float(item["recall"]),
            "mAP50": float(item["mAP50"]),
            "mAP75": float(item["mAP75"]),
            "mAP50-95": float(item["mAP50_95"]),
            "APS": float(item["APS"]),
            "APM": float(item["APM"]),
            "APL": float(item["APL"]),
            "Params": baseline_complexity.get("params"),
            "GFLOPs": baseline_complexity.get("gflops_640"),
        }
        rows.append(row)
    return rows, str(baseline_path)


def durc_row(root, run_id, stage, seed, complexity):
    name = f"durc_{run_id}_{stage.lower()}_seed{seed}"
    test_dir = root / "runs/test" / name
    summary = read_json(test_dir / "summary_metrics.json")
    scale = summary.get("scale_metrics_percent", {})
    row = {
        "stage": stage,
        "repeat": None,
        "seed": seed,
        "config": MODEL_FILES[stage],
        "test_output": str(test_dir),
        "Params": complexity.get("params"),
        "GFLOPs": complexity.get("gflops_640"),
    }
    row.update(
        {label: float(summary["metrics"][key]) for label, key in METRIC_KEYS.items()}
    )
    row.update({label: float(scale[label]) / 100.0 for label in SCALE_KEYS})
    return row


def aggregate_rows(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["stage"], []).append(row)
    aggregate = {}
    for stage, stage_rows in grouped.items():
        metrics = {}
        for key in RESULT_KEYS:
            values = [float(row[key]) for row in stage_rows if row.get(key) is not None]
            metrics[key] = (
                {
                    "mean": statistics.fmean(values),
                    "population_std": statistics.pstdev(values),
                    "min": min(values),
                    "max": max(values),
                }
                if values
                else None
            )
        aggregate[stage] = {"count": len(stage_rows), "metrics": metrics}
    return aggregate


def main():
    args = parse_args()
    root = args.root.resolve()
    state_dir = root / "runs/durc_ablation"
    complexity_payload = read_json(state_dir / "complexity.json")
    rows, baseline_source = baseline_rows(root, complexity_payload.get("baseline", {}))
    for stage in args.stages:
        for seed in args.seeds:
            rows.append(
                durc_row(
                    root,
                    args.run_id,
                    stage,
                    seed,
                    complexity_payload["experiments"][stage],
                )
            )

    payload = {
        "run_id": args.run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "units": {key: "fraction" for key in (*METRIC_KEYS, *SCALE_KEYS)}
        | {"Params": "count", "GFLOPs": "GFLOPs@640"},
        "stage_order": ["A0", *STAGE_ORDER],
        "baseline_source": baseline_source,
        "rows": rows,
        "aggregate": aggregate_rows(rows),
    }

    state_dir.mkdir(parents=True, exist_ok=True)
    json_path = state_dir / f"durc_{args.run_id}_summary.json"
    csv_path = state_dir / f"durc_{args.run_id}_summary.csv"
    temporary = json_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(json_path)

    fields = ["stage", "repeat", "seed", "config", *RESULT_KEYS, "test_output"]
    temporary_csv = csv_path.with_suffix(".csv.tmp")
    with temporary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary_csv.replace(csv_path)
    print(json_path)
    print(csv_path)


if __name__ == "__main__":
    main()
