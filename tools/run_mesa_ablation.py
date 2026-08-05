#!/usr/bin/env python3
"""Train one deterministic MESA-YOLOv13 ablation seed on URPC2020half."""

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
    "m0": ("yolov13-mesa-m0-a3.yaml", "M0 / CARM-A3：CMRF-C4 + MACR"),
    "m1": ("yolov13-mesa-m1-hema.yaml", "M1 / A3 + HEMA"),
    "m2": ("yolov13-mesa-m2-gisr.yaml", "M2 / A3 + GISR"),
    "m3": ("yolov13-mesa-m3-dmbr.yaml", "M3 / A3 + DMBR"),
    "m4": ("yolov13-mesa-m4-hema-gisr.yaml", "M4 / A3 + HEMA + GISR"),
    "m5": ("yolov13-mesa-m5-hema-dmbr.yaml", "M5 / A3 + HEMA + DMBR"),
    "m6": ("yolov13-mesa-m6-gisr-dmbr.yaml", "M6 / A3 + GISR + DMBR"),
    "m7": ("yolov13-mesa-m7-full.yaml", "M7 / 完整 MESA：HEMA + GISR + DMBR"),
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def best_epoch(path: Path) -> int | None:
    if not path.is_file():
        return None
    with path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        return None
    best = max(rows, key=lambda row: float(row.get("metrics/mAP50-95(B)", "-inf")))
    return int(float(best["epoch"]))


def main() -> None:
    args = parse_args()
    root, data = args.root.resolve(), args.data.resolve()
    yaml_name, structure = STAGES[args.stage]
    model_yaml = root / "ultralytics/cfg/models/v13/mesa_ablation" / yaml_name
    pretrained, output = root / "yolov13n.pt", root / "runs/train" / args.name
    for path, label in ((data, "dataset YAML"), (model_yaml, "model YAML"), (pretrained, "pretrained weights")):
        if not path.is_file():
            raise FileNotFoundError(f"MESA {label} unavailable: {path}")
    if output.exists():
        raise FileExistsError(f"refusing to reuse output: {output}")
    os.environ["WANDB_DISABLED"], os.environ["PIN_MEMORY"] = "true", "false"
    sys.path.insert(0, str(root)) if str(root) not in sys.path else None
    import ultralytics
    from ultralytics import YOLO

    try:
        Path(ultralytics.__file__).resolve().relative_to(root)
    except ValueError as error:
        raise ImportError(f"MESA worker resolved external Ultralytics: {ultralytics.__file__}") from error
    model = YOLO(str(model_yaml))
    model.load(str(pretrained))
    if not model.ckpt:
        raise RuntimeError("YOLO.load() did not retain a checkpoint for Trainer")
    started = perf_counter()
    model.train(
        data=str(data), epochs=args.epochs, patience=args.patience, batch=16, imgsz=640, workers=2,
        amp=False, deterministic=True, plots=False, seed=args.seed, resume=False, device=0,
        project=str(root / "runs/train"), name=args.name, exist_ok=False,
    )
    best = output / "weights/best.pt"
    if not best.is_file():
        raise FileNotFoundError(f"MESA training finished without best.pt: {best}")
    payload = {
        "name": args.name, "stage": args.stage.upper(), "seed": args.seed, "structure": structure,
        "model_yaml": str(model_yaml), "dataset": str(data), "weights": str(best), "weights_sha256": sha256(best),
        "initialization": {"method": "YOLO.load", "pretrained": str(pretrained), "pretrained_sha256": sha256(pretrained), "trainer_receives_loaded_model": bool(model.ckpt)},
        "best_epoch": best_epoch(output / "results.csv"), "training_seconds": perf_counter() - started,
        "settings": {"epochs": args.epochs, "patience": args.patience, "batch": 16, "imgsz": 640, "workers": 2, "amp": False, "deterministic": True, "plots": False, "resume": False, "device": 0},
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    (root / "runs/train" / f"{args.name}.train.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
