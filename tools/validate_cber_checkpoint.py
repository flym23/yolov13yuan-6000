"""Audit SCPG-H3 checkpoint migration into the CBER N4 architecture."""

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
    "model.4.detail_router.low_proj.",
    "model.4.detail_router.contrast_proj.",
    "model.4.detail_router.route_gate.",
    "model.4.detail_router.micro_gate.",
    "model.4.detail_router.out_proj.",
    "model.4.detail_router.alpha_raw",
    "model.4.geometry.",
    "model.5.",
    "model.32.",
)
ALLOWED_NEW_PREFIXES = (
    "model.4.detail_router.spatial_compatibility.",
    "model.4.detail_router.channel_compatibility.",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT_DIR)
    parser.add_argument("--weights", type=Path, required=True, help="Existing SCPG-H3 checkpoint.")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root, weights = args.root.resolve(), args.weights.resolve()
    if not weights.is_file():
        raise FileNotFoundError(weights)
    source = YOLO(str(weights)).model.state_dict()
    target_cfg = root / "ultralytics" / "cfg" / "models" / "v13" / "yolov13-cber-n4-full.yaml"
    target = YOLO(str(target_cfg)).model.state_dict()
    matched = {key for key, value in target.items() if key in source and source[key].shape == value.shape}
    missing_required = [prefix for prefix in REQUIRED_PREFIXES if not any(key.startswith(prefix) for key in matched)]
    unexpected = [
        key
        for key in target
        if key not in matched and not key.startswith(ALLOWED_NEW_PREFIXES)
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
    print(f"CBER SCPG-H3 checkpoint migration verification passed: {args.output}")


if __name__ == "__main__":
    main()
