#!/usr/bin/env python3
"""Summarize MESA test JSONs and retain L0/P0 references without overclaiming n=3."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

from run_mesa_ablation import STAGES


METRICS = {"P": "metrics/precision(B)", "R": "metrics/recall(B)", "mAP50": "metrics/mAP50(B)", "mAP75": "metrics/mAP75(B)", "mAP50-95": "metrics/mAP50-95(B)"}
SCALES = ("APS", "APM", "APL")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stages", nargs="+", choices=tuple(STAGES), required=True)
    parser.add_argument("--l0-reference", type=Path, required=True)
    parser.add_argument("--p0-reference", type=Path, required=True)
    return parser.parse_args()


def summary(values: list[float]) -> dict[str, float]:
    return {"mean": statistics.mean(values), "std": statistics.stdev(values) if len(values) > 1 else 0.0, "min": min(values), "max": max(values)}


def reference(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {"path": str(path), "structure": payload.get("structure"), "n": payload.get("n"), "metrics_percent": payload.get("metrics_percent"), "runs": payload.get("runs", [])}


def main() -> None:
    args, root = parse_args(), None
    root = args.root.resolve()
    rows, stages = [], {}
    for stage in args.stages:
        filename, structure = STAGES[stage]
        stage_rows = []
        for seed in (0, 1, 2):
            name = f"{args.run_id}_{stage}_seed{seed}"
            test_path, train_path = root / "runs/test" / name / "summary_metrics.json", root / "runs/train" / f"{name}.train.json"
            if not test_path.is_file() or not train_path.is_file():
                raise FileNotFoundError(f"missing MESA seed record: {test_path} or {train_path}")
            test, train = json.loads(test_path.read_text(encoding="utf-8")), json.loads(train_path.read_text(encoding="utf-8"))
            row = {"stage": stage, "seed": seed, "name": name, "structure": structure, "model_yaml": filename, "weights": test["weights"], "Params": int(test["model"]["parameters"]), "GFLOPs": float(test["model"]["gflops"]), "Best_epoch": train["best_epoch"], "Training_seconds": train["training_seconds"]}
            row.update({label: float(test["metrics"][key]) * 100.0 for label, key in METRICS.items()})
            row.update({label: float(test["scale_metrics_percent"][label]) for label in SCALES})
            rows.append(row)
            stage_rows.append(row)
        stages[stage] = {"structure": structure, "model_yaml": filename, "n": 3, "runs": stage_rows, "metrics_percent": {metric: summary([row[metric] for row in stage_rows]) for metric in (*METRICS, *SCALES)}, "Params": stage_rows[0]["Params"], "GFLOPs": stage_rows[0]["GFLOPs"]}
    m0 = {row["seed"]: row for row in stages["m0"]["runs"]}
    for stage in stages.values():
        stage["vs_same_seed_m0_percent_points"] = {metric: summary([row[metric] - m0[row["seed"]][metric] for row in stage["runs"]]) for metric in (*METRICS, *SCALES)}
        stage["same_seed_m0_map50_95_wins"] = sum(row["mAP50-95"] > m0[row["seed"]]["mAP50-95"] for row in stage["runs"])
    output_dir = root / "runs/test/mesa_ablation"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "MESA_AllSeed_Results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    output = output_dir / "MESA_AllSeed_Summary.json"
    payload = {"run_id": args.run_id, "dataset": "/home/room305/ZZF/URPC2020half/data.yaml", "settings": {"epochs": 300, "patience": 40, "batch": 16, "imgsz": 640, "device": 0, "workers": 2, "amp": False, "deterministic": True, "plots": False, "parallel_seed_processes": 3}, "references": {"L0": reference(args.l0_reference), "P0": reference(args.p0_reference)}, "stages": stages, "csv": str(csv_path), "note": "n=3 仅报告均值、样本标准差、范围和同 seed 方向，不宣称统计显著。"}
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
