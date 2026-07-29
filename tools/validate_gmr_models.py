#!/usr/bin/env python3
"""Build and contract-check every GMR factorial configuration before launch."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from gmr_experiments import ALL_STAGES, MODEL_FILES  # noqa: E402
from ultralytics import YOLO  # noqa: E402
from ultralytics.nn.modules import GIMRDetect, MCASUp, SCPGDSC3k2  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT_DIR)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root, device = args.root.resolve(), torch.device(args.device)
    expected = {
        "r0": (False, False, False), "r1": (True, False, False), "r2": (False, True, False), "r3": (False, False, True),
        "r4": (True, True, False), "r5": (True, False, True), "r6": (False, True, True), "r7": (True, True, True),
    }
    model_dir = root / "ultralytics" / "cfg" / "models" / "v13"
    for stage in ALL_STAGES:
        model = YOLO(str(model_dir / MODEL_FILES[stage])).model
        scgp, mcas, gimr = expected[stage]
        assert len(model.model) == 33 and isinstance(model.model[4], SCPGDSC3k2) is scgp
        assert isinstance(model.model[15], MCASUp) is mcas and isinstance(model.model[32], GIMRDetect) is gimr
        assert list(model.model[32].f) == [23, 27, 31]
        assert torch.allclose(model.model[32].stride.cpu(), torch.tensor([8.0, 16.0, 32.0]))
    for scale in ("n", "s", "l", "x"):
        model = YOLO(str(model_dir / f"yolov13{scale}-gmr.yaml")).model.to(device).eval()
        assert isinstance(model.model[4], SCPGDSC3k2) and isinstance(model.model[15], MCASUp) and isinstance(model.model[32], GIMRDetect)
        if scale == "n":
            with torch.no_grad():
                assert torch.isfinite(model(torch.randn(1, 3, args.imgsz, args.imgsz, device=device))[0]).all()
                assert torch.isfinite(model(torch.randn(1, 3, args.imgsz + 32, args.imgsz + 32, device=device))[0]).all()
            model.fuse(verbose=False)
            with torch.no_grad():
                assert torch.isfinite(model(torch.randn(1, 3, args.imgsz, args.imgsz, device=device))[0]).all()
    print("GMR parser, layer topology, dynamic shape, stride, and fuse verification passed.")


if __name__ == "__main__":
    main()
