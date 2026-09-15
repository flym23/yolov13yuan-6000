#!/usr/bin/env python3
"""Train exactly one DCRQ ablation seed with the fixed URPC2019 protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter


STAGES = {
    "B0_L0": "yolov13n-dcrq-b0.yaml",
    "B1_D": "yolov13n-dcrq-b1.yaml",
    "B2_D_RPQC": "yolov13n-dcrq-b2.yaml",
    "B3_DAWRB_D_RPQC": "yolov13n-dcrq-b3.yaml",
    "B4_DCRQ_FULL": "yolov13n-dcrq-b4.yaml",
}
FIXED_SETTINGS = {
    "epochs": 160,
    "batch": 16,
    "imgsz": 640,
    "device": 0,
    "workers": 2,
    "amp": False,
    "deterministic": True,
    "plots": False,
    "patience": "not set",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--stage", choices=tuple(STAGES), required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--data", type=Path, required=True)
    return parser.parse_args()


def pretrained_transfer(model, pretrained: Path) -> dict:
    """Record every compatible and incompatible key before invoking the supported YOLO.load path."""
    from ultralytics.nn.tasks import torch_safe_load

    checkpoint, _ = torch_safe_load(str(pretrained))
    source_model = checkpoint.get("ema") or checkpoint["model"]
    source = source_model.float().state_dict()
    target = model.model.state_dict()
    transferred = sorted(key for key, value in source.items() if key in target and value.shape == target[key].shape)
    missing = sorted(key for key in target if key not in source or source[key].shape != target[key].shape)
    unexpected = sorted(key for key in source if key not in target or source[key].shape != target[key].shape)
    return {
        "method": "YOLO.load",
        "pretrained": str(pretrained),
        "pretrained_sha256": sha256(pretrained),
        "source_parameter_count": len(source),
        "target_parameter_count": len(target),
        "transferred_parameter_count": len(transferred),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
    }


def main() -> None:
    args = parse_args()
    root, run_root, data = args.root.resolve(), args.run_root.resolve(), args.data.resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import ultralytics

    try:
        Path(ultralytics.__file__).resolve().relative_to(root)
    except ValueError as error:
        raise ImportError(f"DCRQ worker resolved external Ultralytics: {ultralytics.__file__}") from error
    from ultralytics import YOLO

    model_yaml = root / "ultralytics" / "cfg" / "models" / "v13" / STAGES[args.stage]
    pretrained = root / "yolov13n.pt"
    output_dir = run_root / "train" / args.stage / f"seed{args.seed}"
    for path, label in ((data, "dataset YAML"), (model_yaml, "model YAML"), (pretrained, "yolov13n.pt")):
        if not path.is_file():
            raise FileNotFoundError(f"DCRQ {label} is unavailable: {path}")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing DCRQ output: {output_dir}")

    os.environ["WANDB_DISABLED"] = "true"
    os.environ["PIN_MEMORY"] = "false"
    model = YOLO(str(model_yaml))
    transfer = pretrained_transfer(model, pretrained)
    model.load(str(pretrained))
    if not model.ckpt:
        raise RuntimeError("YOLO.load() did not retain a checkpoint for the Trainer.")
    atomic_json(
        output_dir.parent / f"seed{args.seed}.pretrained_load.json",
        {"stage": args.stage, "seed": args.seed, "model_yaml": str(model_yaml), **transfer},
    )

    started = perf_counter()
    model.train(
        data=str(data),
        epochs=160,
        batch=16,
        imgsz=640,
        device=0,
        workers=2,
        amp=False,
        deterministic=True,
        plots=False,
        seed=args.seed,
        resume=False,
        project=str(output_dir.parent),
        name=output_dir.name,
        exist_ok=False,
    )
    best = output_dir / "weights" / "best.pt"
    if not best.is_file():
        raise FileNotFoundError(f"DCRQ training finished without best.pt: {best}")
    atomic_json(
        output_dir / "train_complete.json",
        {
            "stage": args.stage,
            "seed": args.seed,
            "dataset": str(data),
            "model_yaml": str(model_yaml),
            "weights": str(best),
            "weights_sha256": sha256(best),
            "settings": FIXED_SETTINGS,
            "training_seconds": perf_counter() - started,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        },
    )


if __name__ == "__main__":
    main()
