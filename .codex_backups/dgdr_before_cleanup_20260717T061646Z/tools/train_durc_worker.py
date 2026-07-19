#!/usr/bin/env python3
"""Train one reviewed DURC ablation configuration and one deterministic seed."""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from durc_experiments import MODEL_FILES  # noqa: E402
from ultralytics import YOLO  # noqa: E402


def stage_name(value):
    value = value.upper()
    if value not in MODEL_FILES:
        raise argparse.ArgumentTypeError(
            f"stage must be one of {', '.join(MODEL_FILES)}"
        )
    return value


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stage", type=stage_name, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    return parser.parse_args()


def main():
    args = parse_args()
    root = args.root.resolve()
    os.environ["WANDB_DISABLED"] = "true"
    model_cfg = root / "ultralytics/cfg/models/v13" / MODEL_FILES[args.stage]
    data_yaml = root / "data_durc.yaml"
    pretrained = root / "yolov13n.pt"
    output_dir = root / "runs/train" / args.name
    for path in (model_cfg, data_yaml, pretrained):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output_dir.exists():
        raise FileExistsError(
            f"refusing to reuse an existing training directory: {output_dir}"
        )

    metadata_path = root / "runs/durc_ablation" / f"{args.name}.train.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "stage": args.stage,
        "config": str(model_cfg),
        "seed": args.seed,
        "epochs": args.epochs,
        "amp": False,
        "dataloader_workers": 2,
        "label_id_mapping": "URPC 1..4 -> YOLO 0..3 (DURC-private label view)",
        "deterministic": True,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    model = YOLO(str(model_cfg))
    model.load(str(pretrained))
    model.train(
        data=str(data_yaml),
        imgsz=640,
        batch=16,
        epochs=args.epochs,
        optimizer="auto",
        lr0=0.01,
        lrf=0.01,
        close_mosaic=5,
        patience=40,
        workers=2,
        device=0,
        amp=False,
        deterministic=True,
        plots=False,
        seed=args.seed,
        project=str(root / "runs/train"),
        name=args.name,
        exist_ok=False,
    )

    metadata["completed_at"] = datetime.now(timezone.utc).isoformat()
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
