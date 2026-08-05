#!/usr/bin/env python3
"""Verify that an A3 checkpoint transfers only the intended new MESA branch keys."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--a3-weights", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from ultralytics import YOLO

    a3_yaml = root / "ultralytics/cfg/models/v13/mesa_ablation/yolov13-mesa-m0-a3.yaml"
    mesa_yaml = root / "ultralytics/cfg/models/v13/yolov13-mesa.yaml"
    a3, mesa = YOLO(str(a3_yaml)), YOLO(str(mesa_yaml))
    a3.load(str(args.a3_weights))
    result = mesa.model.load_state_dict(a3.model.state_dict(), strict=False)
    allowed = ("model.21.mass_allocator.", "model.21.scale_reactivator.", "model.21.boundary_refiner.")
    if result.unexpected_keys or any(not key.startswith(allowed) for key in result.missing_keys):
        raise AssertionError(f"unexpected={result.unexpected_keys}; invalid_missing={result.missing_keys}")
    print({"missing_keys": result.missing_keys, "unexpected_keys": result.unexpected_keys})


if __name__ == "__main__":
    main()
