#!/usr/bin/env python3
"""Train one deterministic SEAL-YOLOv13 ablation seed on URPC2019."""

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
    "s0": ("yolov13-seal-s0-carm-a3.yaml", "S0 / CARM-A3 同批次基线（CMRF + MACR）"),
    "s1": ("yolov13-seal-s1-dsrf.yaml", "S1 / CARM-A3 + DSRF 浅层退化分离可靠性滤波"),
    "s2": ("yolov13-seal-s2-sarb.yaml", "S2 / CARM-A3 + SARB P4 语义一致性蓄水桥"),
    "s3": ("yolov13-seal-s3-qad.yaml", "S3 / CARM-A3 + QAD P3 质量对齐任务解耦检测头"),
    "s4": ("yolov13-seal-s4-dsrf-sarb.yaml", "S4 / CARM-A3 + DSRF + SARB"),
    "s5": ("yolov13-seal-s5-dsrf-qad.yaml", "S5 / CARM-A3 + DSRF + QAD"),
    "s6": ("yolov13-seal-s6-sarb-qad.yaml", "S6 / CARM-A3 + SARB + QAD"),
    "s7": ("yolov13-seal-s7-full.yaml", "S7 / 完整 SEAL-YOLOv13（DSRF + SARB + QAD）"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    pretrained = root / "yolov13n.pt"
    yaml_name, structure = STAGES[args.stage]
    model_yaml = root / "ultralytics/cfg/models/v13/seal_ablation" / yaml_name
    output_dir = root / "runs/train" / args.name
    for path, label in ((data, "dataset YAML"), (model_yaml, "model YAML"), (pretrained, "YOLOv13n pretrained weights")):
        if not path.is_file():
            raise FileNotFoundError(f"SEAL {label} unavailable: {path}")
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse SEAL output: {output_dir}")

    os.environ["WANDB_DISABLED"] = "true"
    os.environ["PIN_MEMORY"] = "false"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import ultralytics

    try:
        Path(ultralytics.__file__).resolve().relative_to(root)
    except ValueError as error:
        raise ImportError(f"SEAL worker resolved external Ultralytics: {ultralytics.__file__}") from error
    from ultralytics import YOLO

    # This is the current CARM initialization path: YOLO.load() both transfers the
    # matching YOLOv13n tensors and preserves model.ckpt for the Trainer.
    model = YOLO(str(model_yaml))
    model.load(str(pretrained))
    if not model.ckpt:
        raise RuntimeError("YOLO.load() did not retain a checkpoint for Trainer")
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
        project=str(root / "runs/train"), name=args.name, exist_ok=False,
    )
    elapsed_seconds = perf_counter() - started
    best = output_dir / "weights/best.pt"
    if not best.is_file():
        raise FileNotFoundError(f"SEAL training finished without best.pt: {best}")
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
    (root / "runs/train" / f"{args.name}.train.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
