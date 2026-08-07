#!/usr/bin/env python3
"""Aggregate the 3-seed SEAL URPC2019 test outputs without overstating n=3 evidence."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any

from run_seal_ablation import STAGES


METRICS = {
    "P": "metrics/precision(B)",
    "R": "metrics/recall(B)",
    "mAP50": "metrics/mAP50(B)",
    "mAP75": "metrics/mAP75(B)",
    "mAP50-95": "metrics/mAP50-95(B)",
}
SCALE_METRICS = ("APS", "APM", "APL")
ALL_METRICS = tuple(METRICS) + SCALE_METRICS


def statistics_for(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stages", nargs="+", choices=tuple(STAGES), required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    rows: list[dict[str, Any]] = []
    for stage in args.stages:
        yaml_name, structure = STAGES[stage]
        for seed in (0, 1, 2):
            name = f"{args.run_id}_{stage}_seed{seed}"
            test_path = root / "runs/test" / name / "summary_metrics.json"
            scale_path = root / "runs/test" / name / "scale_ap_metrics.json"
            train_path = root / "runs/train" / f"{name}.train.json"
            if not test_path.is_file() or not scale_path.is_file() or not train_path.is_file():
                raise FileNotFoundError(f"missing SEAL seed record for {name}")
            test = json.loads(test_path.read_text(encoding="utf-8"))
            # Keep the scale file as a completion guard; test.py records the normalized
            # APS/APM/APL values in summary_metrics.json for direct comparison.
            json.loads(scale_path.read_text(encoding="utf-8"))
            train = json.loads(train_path.read_text(encoding="utf-8"))
            row = {
                "stage": stage,
                "seed": seed,
                "name": name,
                "structure": structure,
                "model_yaml": str(root / "ultralytics/cfg/models/v13/seal_ablation" / yaml_name),
                "weights": test["weights"],
                "weight_sha256": train["weights_sha256"],
                "Params": int(test["model"].get("parameters", 0)),
                "GFLOPs": float(test["model"].get("gflops", 0.0)),
                "Training_seconds": float(train["training_seconds"]),
                "Best_epoch": train["best_epoch"],
                "Initialization_method": train["initialization"]["method"],
                "Initialization_weights": train["initialization"]["pretrained"],
            }
            row.update({label: float(test["metrics"][key]) * 100.0 for label, key in METRICS.items()})
            row.update({label: float(test.get("scale_metrics_percent", {}).get(label, 0.0)) for label in SCALE_METRICS})
            rows.append(row)

    stages: dict[str, Any] = {}
    s0_by_seed = {row["seed"]: row for row in rows if row["stage"] == "s0"}
    for stage in args.stages:
        stage_rows = [row for row in rows if row["stage"] == stage]
        summary = {
            "structure": STAGES[stage][1],
            "model_yaml": str(root / "ultralytics/cfg/models/v13/seal_ablation" / STAGES[stage][0]),
            "n": len(stage_rows),
            "runs": stage_rows,
            "metrics_percent": {metric: statistics_for([row[metric] for row in stage_rows]) for metric in ALL_METRICS},
            "Params": stage_rows[0]["Params"],
            "GFLOPs": stage_rows[0]["GFLOPs"],
        }
        paired = {
            metric: [row[metric] - s0_by_seed[row["seed"]][metric] for row in stage_rows if row["seed"] in s0_by_seed]
            for metric in ALL_METRICS
        }
        summary["vs_same_seed_s0_percent_points"] = {
            metric: statistics_for(values) for metric, values in paired.items() if values
        }
        summary["same_seed_s0_map50_95_wins"] = sum(
            row["mAP50-95"] > s0_by_seed[row["seed"]]["mAP50-95"]
            for row in stage_rows if row["seed"] in s0_by_seed
        )
        stages[stage] = summary

    output_dir = root / "runs/test/seal_ablation"
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = tuple(rows[0])
    csv_path = output_dir / f"{args.run_id}_AllSeed_Results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    output = {
        "run_id": args.run_id,
        "dataset": "/home/room305/ZZF/URPC2019/data.yaml",
        "settings": {
            "epochs": 300, "patience": 40, "batch": 16, "imgsz": 640, "device": 0, "workers": 2,
            "amp": False, "deterministic": True, "plots": False, "parallel_seed_processes": 3,
        },
        "notes": [
            "S0 是与 S1--S7 同一训练批次的 CARM-A3 对照；差值按同 seed 配对计算。",
            "n=3 仅报告均值、样本标准差、范围和同 seed 胜场，不主张统计显著性。",
            "test.py 未测量延迟和峰值显存，汇总不伪造这两个指标。",
        ],
        "stages": stages,
        "csv": str(csv_path),
    }
    output_path = output_dir / f"{args.run_id}_AllSeed_Summary.json"
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
