#!/usr/bin/env python3
"""Reload and structurally audit DURC best/last checkpoints after each training run."""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ultralytics import YOLO  # noqa: E402
from ultralytics.nn.modules.head import Detect, HRCTDetect, SUDLDetect  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = [f"{group}{index}" for group, count in (("A", 3), ("B", 4), ("C", 4)) for index in range(1, count + 1)]
    parser.add_argument("--stage", required=True, choices=stages)
    parser.add_argument("--best", type=Path, required=True)
    parser.add_argument("--last", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--imgsz", type=int, default=160)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def assert_finite(value):
    if torch.is_tensor(value):
        if not torch.isfinite(value).all():
            raise RuntimeError("checkpoint inference contains NaN or Inf")
    elif isinstance(value, (list, tuple)):
        for item in value:
            assert_finite(item)


def expected_head(stage):
    if stage.startswith("A"):
        return Detect
    if stage.startswith("B"):
        return HRCTDetect
    return SUDLDetect


def audit(path, stage, device, imgsz):
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(path)
    model = YOLO(str(path)).model.to(device).eval()
    head = model.model[-1]
    head_class = expected_head(stage)
    if type(head) is not head_class:
        raise RuntimeError(f"{path} head={type(head).__name__}, expected {head_class.__name__}")
    if stage.startswith("C"):
        if "model.32.dfl.project" not in model.state_dict() or "model.32.dfl.conv.weight" in model.state_dict():
            raise RuntimeError(f"{path} has invalid SUDL projection state")
    alpha = {name: parameter for name, parameter in model.named_parameters() if name.endswith("alpha_raw")}
    if not alpha:
        raise RuntimeError(f"{path} has no DURC alpha parameters")
    if not all(torch.isfinite(parameter).all() for parameter in alpha.values()):
        raise RuntimeError(f"{path} has non-finite alpha parameters")
    with torch.no_grad():
        output = model(torch.randn(1, 3, imgsz, imgsz, device=device))
    assert_finite(output)
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "head": type(head).__name__,
        "alpha_parameters": {name: parameter.detach().float().cpu().tolist() for name, parameter in alpha.items()},
    }


def main():
    args = parse_args()
    device = torch.device(args.device)
    records = [
        audit(args.best.resolve(), args.stage, device, args.imgsz),
        audit(args.last.resolve(), args.stage, device, args.imgsz),
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": args.stage,
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "checkpoints": records,
    }
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
