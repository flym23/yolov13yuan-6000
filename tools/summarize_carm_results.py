#!/usr/bin/env python3
"""Aggregate CARM seed-level test summaries without overstating three-seed evidence."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any

from run_carm_ablation import STAGES


METRICS = {
    "P": "metrics/precision(B)",
    "R": "metrics/recall(B)",
    "mAP50": "metrics/mAP50(B)",
    "mAP75": "metrics/mAP75(B)",
    "mAP50-95": "metrics/mAP50-95(B)",
}
SCALE_METRICS = ("APS", "APM", "APL")
ALL_METRICS = tuple(METRICS) + SCALE_METRICS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stages", nargs="+", choices=tuple(STAGES), required=True)
    parser.add_argument("--l0-reference", type=Path, required=True)
    parser.add_argument("--p0-reference", type=Path, required=True)
    parser.add_argument("--t7-reference", type=Path)
    parser.add_argument("--c4-reference", type=Path)
    return parser.parse_args()


def mean_std_min_max(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def load_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def as_percent(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result * 100.0 if abs(result) <= 1.0 else result


def reference_metrics(payload: dict[str, Any] | None) -> dict[str, float]:
    if not payload:
        return {}
    stages = payload.get("stages", {}) if isinstance(payload.get("stages"), dict) else {}
    for stage_name in ("c4", "t7", "p0", "l0"):
        stage = stages.get(stage_name)
        metric_summary = stage.get("metrics_percent", {}) if isinstance(stage, dict) else {}
        if metric_summary:
            return {
                metric: float(values["mean"])
                for metric, values in metric_summary.items()
                if isinstance(values, dict) and "mean" in values and metric in ALL_METRICS
            }
    metrics = payload.get("metrics", {}) if isinstance(payload.get("metrics"), dict) else {}
    scale = payload.get("scale_metrics_percent", {}) if isinstance(payload.get("scale_metrics_percent"), dict) else {}
    result = {label: as_percent(metrics.get(key)) for label, key in METRICS.items()}
    result.update({label: as_percent(scale.get(label)) for label in SCALE_METRICS})
    return {key: value for key, value in result.items() if value is not None}


def reference_record(label: str, path: Path | None) -> dict[str, Any]:
    payload = load_json(path)
    return {
        "label": label,
        "path": str(path) if path else None,
        "available": payload is not None,
        "metrics_percent": reference_metrics(payload),
        "summary": payload,
    }


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    rows: list[dict[str, Any]] = []
    for stage in args.stages:
        _, structure = STAGES[stage]
        for seed in (0, 1, 2):
            name = f"{args.run_id}_{stage}_seed{seed}"
            test_path = root / "runs" / "test" / name / "summary_metrics.json"
            train_path = root / "runs" / "train" / f"{name}.train.json"
            if not test_path.is_file() or not train_path.is_file():
                raise FileNotFoundError(f"missing CARM seed record: {test_path} or {train_path}")
            test_payload = load_json(test_path)
            train_payload = load_json(train_path)
            assert test_payload is not None and train_payload is not None
            initialization = train_payload["initialization"]
            row = {
                "stage": stage,
                "seed": seed,
                "name": name,
                "structure": structure,
                "weights": test_payload["weights"],
                "weight_sha256": train_payload["weights_sha256"],
                "Params": int(test_payload["model"].get("parameters", 0)),
                "GFLOPs": float(test_payload["model"].get("gflops", 0.0)),
                "Latency_ms": None,
                "FPS": None,
                "Peak_GPU_memory_MiB": None,
                "Training_seconds": float(train_payload["training_seconds"]),
                "Best_epoch": train_payload["best_epoch"],
                "Initialization_method": initialization["method"],
                "Initialization_weights": initialization["pretrained"],
                "Initialization_SHA256": initialization["pretrained_sha256"],
                "Trainer_receives_loaded_model": bool(initialization["trainer_receives_loaded_model"]),
            }
            row.update({label: float(test_payload["metrics"][key]) * 100.0 for label, key in METRICS.items()})
            row.update({label: float(test_payload.get("scale_metrics_percent", {}).get(label, 0.0)) for label in SCALE_METRICS})
            rows.append(row)

    summary: dict[str, Any] = {}
    for stage in args.stages:
        stage_rows = [row for row in rows if row["stage"] == stage]
        summary[stage] = {
            "structure": STAGES[stage][1],
            "n": len(stage_rows),
            "runs": stage_rows,
            "metrics_percent": {metric: mean_std_min_max([float(row[metric]) for row in stage_rows]) for metric in ALL_METRICS},
            "Params": stage_rows[0]["Params"],
            "GFLOPs": stage_rows[0]["GFLOPs"],
        }

    a0_rows = {row["seed"]: row for row in rows if row["stage"] == "a0"}
    for stage, stage_summary in summary.items():
        diffs = {metric: [] for metric in ALL_METRICS}
        for row in stage_summary["runs"]:
            if row["seed"] in a0_rows:
                control = a0_rows[row["seed"]]
                for metric in ALL_METRICS:
                    diffs[metric].append(float(row[metric]) - float(control[metric]))
        stage_summary["vs_same_seed_a0_percent_points"] = {
            metric: mean_std_min_max(values) for metric, values in diffs.items() if values
        }
        stage_summary["same_seed_a0_map50_95_wins"] = sum(
            row["mAP50-95"] > a0_rows[row["seed"]]["mAP50-95"]
            for row in stage_summary["runs"] if row["seed"] in a0_rows
        )

    references = {
        "L0": reference_record("L0", args.l0_reference),
        "P0": reference_record("P0", args.p0_reference),
        "T7": reference_record("T7", args.t7_reference),
        "C4": reference_record("C4", args.c4_reference),
    }
    for stage_summary in summary.values():
        stage_means = {metric: stage_summary["metrics_percent"][metric]["mean"] for metric in ALL_METRICS}
        stage_summary["vs_reference_percent_points"] = {
            label: {
                metric: stage_means[metric] - ref_value
                for metric, ref_value in reference["metrics_percent"].items()
                if metric in stage_means
            }
            for label, reference in references.items()
            if reference["available"]
        }

    output_dir = root / "runs" / "test" / "carm_ablation"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "CARM_AllSeed_Results.csv"
    fields = (
        "stage", "seed", "name", "P", "R", "mAP50", "mAP75", "mAP50-95", "APS", "APM", "APL",
        "Params", "GFLOPs", "Latency_ms", "FPS", "Peak_GPU_memory_MiB", "Training_seconds", "Best_epoch",
        "Initialization_method", "Initialization_weights", "Initialization_SHA256",
        "Trainer_receives_loaded_model", "weight_sha256",
    )
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)

    result = {
        "run_id": args.run_id,
        "dataset": "/home/room305/ZZF/URPC2020half/data.yaml",
        "settings": {
            "epochs": 300, "patience": 40, "batch": 16, "imgsz": 640, "device": 0, "workers": 2,
            "amp": False, "deterministic": True, "plots": False, "parallel_seed_processes": 3,
        },
        "notes": [
            "每组固定执行 seed=0,1,2；汇总提供均值、样本标准差、最小值和最大值，不将 n=3 表述为统计显著。",
            "当前 test.py 不输出延迟、FPS 和峰值显存；这些字段显式保留为 null，避免伪造性能数据。",
        ],
        "references": references,
        "stages": summary,
        "csv": str(csv_path),
    }
    output_path = output_dir / "CARM_AllSeed_Summary.json"
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
