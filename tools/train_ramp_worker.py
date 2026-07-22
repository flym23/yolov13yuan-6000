#!/usr/bin/env python3
"""Train one deterministic RAMP K1--K4 seed on URPC2020half."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

os.environ["PIN_MEMORY"] = "false"

from ramp_experiments import MODEL_FILES, STRUCTURES, resolve_model  # noqa: E402
from ultralytics import YOLO  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stage", choices=tuple(MODEL_FILES), required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=40)
    return parser.parse_args()


def validate_data_yaml(path: Path) -> None:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    names = data.get("names", {})
    names = dict(enumerate(names)) if isinstance(names, list) else {int(key): value for key, value in names.items()}
    if set(names) != {0, 1, 2, 3}:
        raise ValueError(f"URPC2020half must expose class IDs 0..3, got {sorted(names)}")


def main() -> None:
    args = parse_args()
    root, data_yaml = args.root.resolve(), args.data.resolve()
    model_cfg, pretrained = resolve_model(root, args.stage), root / "yolov13n.pt"
    output_dir = root / "runs" / "train" / args.name
    for path in (model_cfg, data_yaml, pretrained):
        if not path.is_file():
            raise FileNotFoundError(path)
    validate_data_yaml(data_yaml)
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse training directory: {output_dir}")

    os.environ["WANDB_DISABLED"] = "true"
    metadata_path = root / "runs" / "train" / f"{args.name}.train.json"
    metadata = {
        "stage": args.stage,
        "structure": STRUCTURES[args.stage],
        "config": str(model_cfg),
        "data": str(data_yaml),
        "seed": args.seed,
        "epochs": args.epochs,
        "patience": args.patience,
        "amp": False,
        "dataloader_workers": 2,
        "plots": False,
        "deterministic": True,
        "pin_memory": False,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    model = YOLO(str(model_cfg))
    model.load(str(pretrained))
    model.train(
        data=str(data_yaml), imgsz=640, batch=16, epochs=args.epochs, optimizer="auto", lr0=0.01, lrf=0.01,
        close_mosaic=5, patience=args.patience, workers=2, device=0, amp=False, deterministic=True, plots=False,
        seed=args.seed, project=str(root / "runs" / "train"), name=args.name, exist_ok=False,
    )
    best = output_dir / "weights" / "best.pt"
    if not best.is_file():
        raise FileNotFoundError(f"training completed without best checkpoint: {best}")
    metadata.update(completed_at=datetime.now(timezone.utc).isoformat(), best_weights=str(best))
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
