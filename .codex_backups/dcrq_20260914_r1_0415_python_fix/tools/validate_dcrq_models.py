#!/usr/bin/env python3
"""Build and forward-check DCRQ B2/B3/B4 from outside the repository working directory."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    os.chdir("/tmp")
    sys.path.insert(0, str(root))
    import torch
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.nn.modules.head import RPQUDQDetect

    Path(ultralytics.__file__).resolve().relative_to(root)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for stage in ("b2", "b3", "b4"):
        model = YOLO(str(root / "ultralytics" / "cfg" / "models" / "v13" / f"yolov13n-dcrq-{stage}.yaml")).model.to(device).eval()
        head = model.model[-1]
        if not isinstance(head, RPQUDQDetect) or head.stride.tolist() != [8.0, 16.0, 32.0]:
            raise RuntimeError(f"DCRQ {stage} head or stride contract failed: {type(head).__name__}, {head.stride.tolist()}")
        with torch.inference_mode():
            prediction, raw = model(torch.randn(1, 3, 640, 640, device=device))
        if prediction.shape[1] != 8 or [feature.shape[-2:] for feature in raw] != [(80, 80), (40, 40), (20, 20)]:
            raise RuntimeError(f"DCRQ {stage} forward contract failed.")
        print(f"{stage}: stride={head.stride.tolist()} raw_shapes={[tuple(feature.shape) for feature in raw]}")


if __name__ == "__main__":
    main()
