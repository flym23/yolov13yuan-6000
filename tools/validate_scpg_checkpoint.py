#!/usr/bin/env python3
"""Audit baseline YOLOv13-n checkpoint migration into the full SCPG H5 model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ultralytics import YOLO  # noqa: E402

REQUIRED_PREFIXES = (
    "model.4.cv1.",
    "model.4.cv2.",
    "model.4.m.",
    "model.32.cv2.",
    "model.32.cv3.",
    "model.32.dfl.",
)
ALLOWED_NEW_PREFIXES = (
    "model.4.detail_router.",
    "model.4.geometry.",
    "model.32.p3_adapter.",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT_DIR)
    parser.add_argument("--weights", type=Path, default=ROOT_DIR / "yolov13n.pt")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root, weights = args.root.resolve(), args.weights.resolve()
    target_cfg = root / "ultralytics" / "cfg" / "models" / "v13" / "yolov13n-scpg-h5-full.yaml"
    if not weights.is_file():
        raise FileNotFoundError(weights)
    source = YOLO(str(weights)).model.state_dict()
    target = YOLO(str(target_cfg)).model.state_dict()
    matched = {key for key, value in target.items() if key in source and source[key].shape == value.shape}
    missing_required = [prefix for prefix in REQUIRED_PREFIXES if not any(key.startswith(prefix) for key in matched)]
    unexpected = [
        key for key in target
        if key not in matched and not key.startswith(ALLOWED_NEW_PREFIXES) and not key.startswith("model.32.cv3.")
    ]
    report = {
        "source_weights": str(weights),
        "target_config": str(target_cfg),
        "source_keys": len(source),
        "target_keys": len(target),
        "matched_keys": len(matched),
        "missing_required_prefixes": missing_required,
        "unexpected_unmatched_target_keys": unexpected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if missing_required or unexpected:
        raise RuntimeError(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"SCPG checkpoint migration verification passed: {args.output}")


if __name__ == "__main__":
    main()

