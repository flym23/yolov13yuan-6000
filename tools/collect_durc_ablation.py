#!/usr/bin/env python3
"""Collect DURC metrics using the DCRA three-seed test-summary schema."""

import argparse
import csv
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

try:  # Supports both ``python tools/...`` and module imports in regression tests.
    from .durc_experiments import MODEL_FILES, STAGE_ORDER
except ImportError:
    from durc_experiments import MODEL_FILES, STAGE_ORDER


METRIC_KEYS = {
    "P": ("metrics", "metrics/precision(B)", 100.0),
    "R": ("metrics", "metrics/recall(B)", 100.0),
    "mAP50": ("metrics", "metrics/mAP50(B)", 100.0),
    "mAP75": ("metrics", "metrics/mAP75(B)", 100.0),
    "mAP50-95": ("metrics", "metrics/mAP50-95(B)", 100.0),
    "APS": ("scale_metrics_percent", "APS", 1.0),
    "APM": ("scale_metrics_percent", "APM", 1.0),
    "APL": ("scale_metrics_percent", "APL", 1.0),
}

STAGE_CONFIGS = {
    "A1": {
        "yaml": MODEL_FILES["A1"],
        "structure": "YOLOv13 + DRSC at P2 (detail feature stage)",
    },
    "A2": {
        "yaml": MODEL_FILES["A2"],
        "structure": "YOLOv13 + DRSC at P3 (mid-level feature stage)",
    },
    "A3": {
        "yaml": MODEL_FILES["A3"],
        "structure": "YOLOv13 + DRSC at P2 and P3",
    },
    "B1": {
        "yaml": MODEL_FILES["B1"],
        "structure": "A3 + HRCT detection calibration at P3",
    },
    "B2": {
        "yaml": MODEL_FILES["B2"],
        "structure": "A3 + HRCT detection calibration at P4",
    },
    "B3": {
        "yaml": MODEL_FILES["B3"],
        "structure": "A3 + HRCT detection calibration at P3 and P4",
    },
    "B4": {
        "yaml": MODEL_FILES["B4"],
        "structure": "A3 + HRCT detection calibration at P3, P4, and P5",
    },
    "C1": {
        "yaml": MODEL_FILES["C1"],
        "structure": "B4 + SUDL with non-uniform DFL projection",
    },
    "C2": {
        "yaml": MODEL_FILES["C2"],
        "structure": "C1 + SUDL soft-label distribution supervision",
    },
    "C3": {
        "yaml": MODEL_FILES["C3"],
        "structure": "C1 + SUDL uncertainty, scale weighting, and quality calibration",
    },
    "C4": {
        "yaml": MODEL_FILES["C4"],
        "structure": "B4 + full SUDL (non-uniform DFL, soft-label, uncertainty calibration)",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stages", nargs="+", choices=STAGE_ORDER, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument(
        "--seed-overrides",
        default="",
        help="Optional stage-specific seeds, e.g. 'C2=1,2;C3=0,1,2'.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def parse_seed_overrides(specification: str) -> dict[str, list[int]]:
    """Parse stage-specific seed lists without weakening the default three-seed policy."""
    if not specification.strip():
        return {}
    overrides = {}
    for item in specification.split(";"):
        stage, separator, seed_text = item.strip().partition("=")
        if not separator or stage not in STAGE_ORDER:
            raise ValueError(f"invalid seed override: {item!r}")
        seeds = [int(value) for value in seed_text.split(",") if value]
        if not seeds or len(seeds) != len(set(seeds)):
            raise ValueError(f"seed override must contain unique seeds: {item!r}")
        overrides[stage] = seeds
    return overrides


def load_seed_metrics(root: Path, run_id: str, stage: str, seed: int) -> dict:
    name = f"durc_{run_id}_{stage.lower()}_seed{seed}"
    path = root / "runs/test" / name / "summary_metrics.json"
    data = read_json(path)
    row = {
        "stage": stage,
        "seed": seed,
        "yaml": STAGE_CONFIGS[stage]["yaml"],
        "structure": STAGE_CONFIGS[stage]["structure"],
        "summary_path": str(path),
    }
    for label, (section, key, scale) in METRIC_KEYS.items():
        row[label] = float(data[section][key]) * scale
    return row


def summarize(rows: list[dict]) -> dict:
    output = {}
    for key in METRIC_KEYS:
        values = [row[key] for row in rows]
        output[key] = {
            "mean": statistics.fmean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "values": values,
            "best": max(values),
            "worst": min(values),
        }
    return output


def write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def stage_payload(run_id: str, stage: str, seeds: list[int], rows: list[dict]) -> dict:
    detail = summarize(rows)
    return {
        "run_id": run_id,
        "stage": stage,
        "config": STAGE_CONFIGS[stage],
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "seeds": seeds,
        "rows_percent": rows,
        "mean": {key: value["mean"] for key, value in detail.items()},
        "std": {key: value["std"] for key, value in detail.items()},
        "detail": detail,
    }


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    test_dir = root / "runs/test"
    test_dir.mkdir(parents=True, exist_ok=True)
    seed_overrides = parse_seed_overrides(args.seed_overrides)
    all_rows = []
    summaries = {}
    for stage in args.stages:
        stage_seeds = seed_overrides.get(stage, args.seeds)
        rows = [
            load_seed_metrics(root, args.run_id, stage, seed) for seed in stage_seeds
        ]
        all_rows.extend(rows)
        payload = stage_payload(args.run_id, stage, stage_seeds, rows)
        summaries[stage] = payload
        write_json_atomic(
            test_dir / f"durc_{args.run_id}_{stage.lower()}_summary.json", payload
        )

    csv_path = test_dir / f"durc_{args.run_id}_ablation.csv"
    fields = ["stage", "seed", "yaml", "structure", *METRIC_KEYS, "summary_path"]
    temporary_csv = csv_path.with_suffix(".csv.tmp")
    with temporary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)
    temporary_csv.replace(csv_path)

    combined = {
        "run_id": args.run_id,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "stage_order": args.stages,
        "csv": str(csv_path),
        "summaries": summaries,
    }
    write_json_atomic(test_dir / f"durc_{args.run_id}_summary.json", combined)
    print(
        json.dumps(
            {
                "csv": str(csv_path),
                "stages": args.stages,
                "seeds": args.seeds,
                "seed_overrides": seed_overrides,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
