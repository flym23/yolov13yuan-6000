#!/usr/bin/env python3
"""Train one deterministic CPCR ablation seed and persist side-branch diagnostics."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
os.environ["PIN_MEMORY"] = "false"

from cpcr_experiments import ALL_STAGES, MODEL_FILES, STRUCTURES, resolve_model  # noqa: E402
from ultralytics import YOLO  # noqa: E402
from ultralytics.nn.modules import CPCRDetect  # noqa: E402


class CPCRDiagnosticsRecorder:
    """Record detached side-branch tensors periodically without touching loss or gradients."""

    def __init__(self, path: Path, interval: int = 100):
        self.path, self.interval, self.step = path, int(interval), 0

    @staticmethod
    def _summary(value: torch.Tensor) -> dict[str, float]:
        value = value.detach().float().flatten()
        return {
            "mean": float(value.mean().item()),
            "variance": float(value.var(unbiased=False).item()),
            "p95": float(torch.quantile(value, 0.95).item()),
            "max": float(value.max().item()),
        }

    def __call__(self, trainer) -> None:
        self.step += 1
        if self.step % self.interval:
            return
        head = trainer.model.model[-1]
        if not isinstance(head, CPCRDetect) or not head.latest_diagnostics:
            return
        diagnostics, recovery = head.latest_diagnostics, head.recovery
        record = {
            "utc": datetime.now(timezone.utc).isoformat(),
            "step": self.step,
            "epoch": int(getattr(trainer, "epoch", -1)) + 1,
            "gain_sigmoid_per_class": [float(item) for item in torch.sigmoid(recovery.gain_raw.detach()).cpu()],
            "candidate_probability": self._summary(torch.sigmoid(diagnostics["candidate_logits"])),
            "class_gate": self._summary(diagnostics["class_gate"]),
            "spatial_prior": self._summary(diagnostics["spatial_prior"]),
            "localization_guard": self._summary(diagnostics["localization_guard"]),
            "delta": self._summary(diagnostics["delta"]),
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stage", choices=ALL_STAGES, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=40)
    return parser.parse_args()


def validate_data_yaml(path: Path) -> None:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    names = payload.get("names", {})
    names = dict(enumerate(names)) if isinstance(names, list) else {int(key): value for key, value in names.items()}
    if set(names) != {0, 1, 2, 3}:
        raise ValueError(f"URPC2020half must expose class IDs 0..3, got {sorted(names)}")


def main() -> None:
    args = parse_args()
    root, data = args.root.resolve(), args.data.resolve()
    model_cfg, pretrained, output_dir = resolve_model(root, args.stage), root / "yolov13n.pt", root / "runs" / "train" / args.name
    for path in (model_cfg, data, pretrained):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse training directory: {output_dir}")
    validate_data_yaml(data)
    os.environ["WANDB_DISABLED"] = "true"
    metadata_path = root / "runs" / "train" / f"{args.name}.train.json"
    diagnostic_path = root / "runs" / "train" / f"{args.name}.cpcr_diagnostics.jsonl"
    metadata = {"stage": args.stage, "structure": STRUCTURES[args.stage], "config": str(model_cfg), "data": str(data), "seed": args.seed,
                "epochs": args.epochs, "patience": args.patience, "amp": False, "dataloader_workers": 2, "plots": False,
                "deterministic": True, "pin_memory": False, "diagnostics": str(diagnostic_path), "started_at": datetime.now(timezone.utc).isoformat()}
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    model = YOLO(str(model_cfg))
    model.load(str(pretrained))
    model.add_callback("on_train_batch_end", CPCRDiagnosticsRecorder(diagnostic_path))
    model.train(data=str(data), imgsz=640, batch=16, epochs=args.epochs, optimizer="auto", lr0=0.01, lrf=0.01, close_mosaic=5,
                patience=args.patience, workers=2, device=0, amp=False, deterministic=True, plots=False, seed=args.seed,
                project=str(root / "runs" / "train"), name=args.name, exist_ok=False)
    best = output_dir / "weights" / "best.pt"
    if not best.is_file():
        raise FileNotFoundError(f"training completed without best checkpoint: {best}")
    metadata.update(completed_at=datetime.now(timezone.utc).isoformat(), best_weights=str(best))
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
