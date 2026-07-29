#!/usr/bin/env python3
"""Train one deterministic CMRF-YOLOv13 ablation seed on URPC2020half."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


STAGES = {
    "c0": ("yolov13-cmrf-c0-h3.yaml", "C0 / SCPG-H3 (SER + CPPG)"),
    "c1": ("yolov13-cmrf-c1-crr.yaml", "C1 / CRR"),
    "c2": ("yolov13-cmrf-c2-smpgr.yaml", "C2 / SMPG-R"),
    "c3": ("yolov13-cmrf-c3-h3-rfa.yaml", "C3 / SCPG-H3 + RFA-Up"),
    "c4": ("yolov13-cmrf-c4-crr-smpg.yaml", "C4 / CRR + SMPG"),
    "c5": ("yolov13-cmrf-c5-full.yaml", "C5 / full CMRF-YOLOv13"),
}


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


def main() -> None:
    args = parse_args()
    root, data = args.root.resolve(), args.data.resolve()
    yaml_name, structure = STAGES[args.stage]
    model_yaml = root / "ultralytics" / "cfg" / "models" / "v13" / "cmrf_ablation" / yaml_name
    pretrained = root / "yolov13n.pt"
    output_dir = root / "runs" / "train" / args.name
    for path, label in ((data, "dataset YAML"), (model_yaml, "model YAML"), (pretrained, "pretrained weights")):
        if not path.is_file():
            raise FileNotFoundError(f"CMRF {label} unavailable: {path}")
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse training directory: {output_dir}")

    os.environ["WANDB_DISABLED"] = "true"
    os.environ["PIN_MEMORY"] = "false"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import ultralytics

    try:
        Path(ultralytics.__file__).resolve().relative_to(root)
    except ValueError as error:
        raise ImportError(f"CMRF worker resolved external Ultralytics: {ultralytics.__file__}") from error
    from ultralytics import YOLO

    model = YOLO(str(model_yaml))
    model.load(str(pretrained))
    model.train(
        data=str(data), epochs=args.epochs, patience=args.patience, batch=16, imgsz=640, workers=2,
        amp=False, deterministic=True, plots=False, seed=args.seed, resume=False, device=0,
        project=str(root / "runs" / "train"), name=args.name, exist_ok=False,
    )
    best = output_dir / "weights" / "best.pt"
    if not best.is_file():
        raise FileNotFoundError(f"CMRF training finished without best.pt: {best}")
    payload = {
        "name": args.name, "stage": args.stage.upper(), "seed": args.seed, "structure": structure,
        "model_yaml": str(model_yaml), "dataset": str(data), "weights": str(best),
        "settings": {"epochs": args.epochs, "patience": args.patience, "batch": 16, "imgsz": 640, "workers": 2,
                     "amp": False, "deterministic": True, "plots": False, "resume": False, "device": 0},
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    (root / "runs" / "train" / f"{args.name}.train.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
