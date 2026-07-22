#!/usr/bin/env python3
"""Build and contract-check every CPCR model before training is launched."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from cpcr_experiments import ALL_STAGES, MODEL_FILES  # noqa: E402
from ultralytics import YOLO  # noqa: E402
from ultralytics.nn.modules import CPCRDetect, SCPGDSC3k2  # noqa: E402

EXPECTED = {"n": (64, [64, 128, 256]), "s": (128, [128, 256, 512]), "l": (256, [256, 512, 512]), "x": (384, [384, 512, 512])}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT_DIR)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir, device = args.root.resolve() / "ultralytics" / "cfg" / "models" / "v13", torch.device(args.device)
    expected_flags = {"l1": (False, False, False), "l2": (True, False, False), "l3": (True, True, False), "l4": (True, True, True), "l5": (True, True, True)}
    for stage in ALL_STAGES:
        model = YOLO(str(model_dir / MODEL_FILES[stage].replace("yolov13-", "yolov13n-"))).model
        head = model.model[32]
        assert len(model.model) == 33 and list(head.f) == [2, 23, 27, 31] and isinstance(head, CPCRDetect)
        assert head.nl == 3 and torch.allclose(head.stride.cpu(), torch.tensor([8.0, 16.0, 32.0]))
        assert (head.recovery.use_spatial_prior, head.recovery.use_class_gate, head.recovery.use_loc_guard) == expected_flags[stage]
        assert isinstance(model.model[4], SCPGDSC3k2) is (stage == "l5")
    for scale, (c_shallow, detect_channels) in EXPECTED.items():
        model = YOLO(str(model_dir / f"yolov13{scale}-cpcr.yaml")).model.to(device).eval()
        head = model.model[32]
        assert isinstance(head, CPCRDetect) and head.c_shallow == c_shallow and head.nl == 3
        assert [tower[0].conv.in_channels for tower in head.cv2] == detect_channels
        assert torch.allclose(head.stride.cpu(), torch.tensor([8.0, 16.0, 32.0]))
        head.bias_init()
        if scale == "n":
            with torch.no_grad():
                prediction = model(torch.randn(1, 3, args.imgsz, args.imgsz, device=device))[0]
            assert torch.isfinite(prediction).all()
            model.fuse(verbose=False)
            with torch.no_grad():
                fused = model(torch.randn(1, 3, args.imgsz, args.imgsz, device=device))[0]
            assert torch.isfinite(fused).all()
    print("CPCR configuration verification passed for L1--L5 and n/s/l/x.")


if __name__ == "__main__":
    main()
