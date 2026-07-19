#!/usr/bin/env python3
"""Audit original YOLOv13 checkpoint migration into the complete DGDR model."""

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
    "model.3.conv.",
    "model.3.bn.",
    "model.4.cv1.",
    "model.4.cv2.",
    "model.4.m.",
    "model.32.cv2.",
    "model.32.cv3.",
    "model.32.dfl.",
)
ALLOWED_NEW_PREFIXES = (
    "model.3.detail_proj.",
    "model.3.low_proj.",
    "model.3.threshold.",
    "model.3.channel_gate.",
    "model.3.spatial_gate.",
    "model.3.out_proj.",
    "model.3.alpha_raw",
    "model.4.geometry.",
    "model.32.side_refiner.",
    "model.32.side_alpha_raw",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT_DIR)
    parser.add_argument("--weights", type=Path, default=ROOT_DIR / "yolov13n.pt")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root, weights = args.root.resolve(), args.weights.resolve()
    target_cfg = root / "ultralytics" / "cfg" / "models" / "v13" / "yolov13n-dgdr-g7-full.yaml"
    if not weights.is_file():
        raise FileNotFoundError(weights)
    source = YOLO(str(weights)).model.state_dict()
    target = YOLO(str(target_cfg)).model.state_dict()
    matched = {key for key, value in target.items() if key in source and source[key].shape == value.shape}
    missing_required = [
        prefix for prefix in REQUIRED_PREFIXES if not any(key.startswith(prefix) for key in matched)
    ]
    incompatible_classifier = [key for key in target if key not in matched and key.startswith("model.32.cv3.")]
    unexpected_target = [
        key
        for key in target
        if key not in matched
        and not key.startswith(ALLOWED_NEW_PREFIXES)
        and not key.startswith("model.32.cv3.")
    ]
    report = {
        "source_weights": str(weights),
        "target_config": str(target_cfg),
        "source_keys": len(source),
        "target_keys": len(target),
        "matched_keys": len(matched),
        "matched_classifier_keys": sum(key.startswith("model.32.cv3.") for key in matched),
        "reinitialized_classifier_keys": incompatible_classifier,
        "missing_required_prefixes": missing_required,
        "unexpected_unmatched_target_keys": unexpected_target,
    }
    output = args.output or root / "runs" / "dgdr_ablation" / "checkpoint_migration_preflight.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if missing_required or unexpected_target:
        raise RuntimeError(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"DGDR checkpoint migration verification passed: {output}")


if __name__ == "__main__":
    main()
