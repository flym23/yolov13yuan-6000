#!/usr/bin/env python3
"""Train one deterministic CARM-YOLOv13 ablation seed on URPC2020half."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter


STAGES = {
    "a0": ("yolov13-carm-a0-c4.yaml", "A0 / CMRF-C4 同批次基线（无 MOMR、OCAF-Up、MACR）"),
    "a1": ("yolov13-carm-a1-momr.yaml", "A1 / CMRF-C4 + MOMR"),
    "a2": ("yolov13-carm-a2-ocaf.yaml", "A2 / CMRF-C4 + OCAF-Up"),
    "a3": ("yolov13-carm-a3-macr.yaml", "A3 / CMRF-C4 + MACR"),
    "a4": ("yolov13-carm-a4-momr-ocaf.yaml", "A4 / CMRF-C4 + MOMR + OCAF-Up"),
    "a5": ("yolov13-carm-a5-momr-macr.yaml", "A5 / CMRF-C4 + MOMR + MACR"),
    "a6": ("yolov13-carm-a6-ocaf-macr.yaml", "A6 / CMRF-C4 + OCAF-Up + MACR"),
    "a7": ("yolov13-carm-a7-full.yaml", "A7 / 完整 CARM-YOLOv13（MOMR + OCAF-Up + MACR）"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stage", choices=tuple(STAGES), required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=40)
    return parser.parse_args()


def best_epoch(results_csv: Path) -> int | None:
    if not results_csv.is_file():
        return None
    with results_csv.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        return None
    metric = "metrics/mAP50-95(B)"
    best = max(rows, key=lambda row: float(row.get(metric, "-inf")))
    try:
        return int(float(best.get("epoch", "")))
    except (TypeError, ValueError):
        return None


def main() -> None:
    args = parse_args()
    root, data = args.root.resolve(), args.data.resolve()
    pretrained = root / "yolov13n.pt"
    yaml_name, structure = STAGES[args.stage]
    model_yaml = root / "ultralytics" / "cfg" / "models" / "v13" / "carm_ablation" / yaml_name
    output_dir = root / "runs" / "train" / args.name
    for path, label in ((data, "dataset YAML"), (model_yaml, "model YAML"), (pretrained, "YOLOv13n pretrained weights")):
        if not path.is_file():
            raise FileNotFoundError(f"CARM {label} unavailable: {path}")
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse CARM output: {output_dir}")

    os.environ["WANDB_DISABLED"] = "true"
    os.environ["PIN_MEMORY"] = "false"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import ultralytics

    try:
        Path(ultralytics.__file__).resolve().relative_to(root)
    except ValueError as error:
        raise ImportError(f"CARM worker resolved external Ultralytics: {ultralytics.__file__}") from error
    from ultralytics import YOLO

    model = YOLO(str(model_yaml))
    # Keep the same public loading path as the CMRF workers. Besides loading matching
    # tensors, YOLO.load() sets model.ckpt so Model.train() forwards this model to Trainer.
    model.load(str(pretrained))
    initialization = {
        "stage": args.stage,
        "seed": args.seed,
        "model_yaml": str(model_yaml),
        "method": "YOLO.load",
        "pretrained": str(pretrained),
        "pretrained_sha256": sha256(pretrained),
        "trainer_receives_loaded_model": bool(model.ckpt),
    }

    started = perf_counter()
    model.train(
        data=str(data), epochs=args.epochs, patience=args.patience, batch=16, imgsz=640, workers=2,
        amp=False, deterministic=True, plots=False, seed=args.seed, resume=False, device=0,
        project=str(root / "runs" / "train"), name=args.name, exist_ok=False,
    )
    elapsed_seconds = perf_counter() - started
    best = output_dir / "weights" / "best.pt"
    if not best.is_file():
        raise FileNotFoundError(f"CARM training finished without best.pt: {best}")
    payload = {
        "name": args.name,
        "stage": args.stage.upper(),
        "seed": args.seed,
        "structure": structure,
        "model_yaml": str(model_yaml),
        "dataset": str(data),
        "weights": str(best),
        "weights_sha256": sha256(best),
        "initialization": initialization,
        "best_epoch": best_epoch(output_dir / "results.csv"),
        "training_seconds": elapsed_seconds,
        "settings": {
            "epochs": args.epochs, "patience": args.patience, "batch": 16, "imgsz": 640, "workers": 2,
            "amp": False, "deterministic": True, "plots": False, "resume": False, "device": 0,
        },
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    (root / "runs" / "train" / f"{args.name}.train.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
