#!/usr/bin/env python3
"""Build, forward-check, and fuse-check SCPG H1--H5 before training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ultralytics import YOLO  # noqa: E402
from ultralytics.nn.modules import P3DecoupledDetect, SCPGDSC3k2  # noqa: E402

EXPECTED = {
    "n": (64, 64, 128, [64, 128, 256], 1, False),
    "s": (128, 128, 256, [128, 256, 512], 1, False),
    "l": (256, 256, 512, [256, 512, 512], 2, True),
    "x": (384, 384, 512, [384, 512, 512], 2, True),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT_DIR)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = args.root.resolve() / "ultralytics" / "cfg" / "models" / "v13"
    device = torch.device(args.device)
    for stage in ("h1-ser", "h2-cppg", "h3-ser-cppg", "h4-p3tda", "h5-full"):
        model = YOLO(str(model_dir / f"yolov13n-scpg-{stage}.yaml")).model
        assert len(model.model) == 33 and list(model.model[32].f) == [23, 27, 31]

    for scale, expected in EXPECTED.items():
        c_shallow, c_p3, c_out, head_channels, repeats, dsc3k = expected
        model = YOLO(str(model_dir / f"yolov13{scale}-scpg.yaml")).model.to(device).eval()
        scpg, detect = model.model[4], model.model[32]
        assert isinstance(scpg, SCPGDSC3k2) and isinstance(detect, P3DecoupledDetect)
        assert (scpg.c_shallow, scpg.c_p3, scpg.c2) == (c_shallow, c_p3, c_out)
        assert len(scpg.m) == repeats and scpg.dsc3k_enabled is dsc3k
        assert [tower[0].conv.in_channels for tower in detect.cv2] == head_channels
        assert torch.allclose(detect.stride.cpu(), torch.tensor([8.0, 16.0, 32.0]))
        if scale == "n":
            with torch.no_grad():
                prediction = model(torch.randn(1, 3, args.imgsz, args.imgsz, device=device))[0]
            assert torch.isfinite(prediction).all()
            model.fuse(verbose=False)
            with torch.no_grad():
                fused = model(torch.randn(1, 3, args.imgsz, args.imgsz, device=device))[0]
            assert torch.isfinite(fused).all()
    print("SCPG configuration verification passed for H1--H5 and n/s/l/x.")


if __name__ == "__main__":
    main()

