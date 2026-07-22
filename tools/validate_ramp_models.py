#!/usr/bin/env python3
"""Build and contract-check RAMP K1--K4 before any training is launched."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ultralytics import YOLO  # noqa: E402
from ultralytics.nn.modules import RAMPDetect, SCPGDSC3k2  # noqa: E402

EXPECTED = {
    "n": (64, 64, [64, 128, 256], 1, False),
    "s": (128, 128, [128, 256, 512], 1, False),
    "l": (256, 256, [256, 512, 512], 2, True),
    "x": (384, 384, [384, 512, 512], 2, True),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT_DIR)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir, device = args.root.resolve() / "ultralytics" / "cfg" / "models" / "v13", torch.device(args.device)
    for stage, layer4_name, ambiguity, channel in (
        ("k1-baseline", "DSC3k2", True, True),
        ("k2-prior", "SCPGDSC3k2", False, False),
        ("k3-ambiguity", "SCPGDSC3k2", True, False),
        ("k4-full", "SCPGDSC3k2", True, True),
    ):
        model = YOLO(str(model_dir / f"yolov13n-ramp-{stage}.yaml")).model
        head = model.model[32]
        assert len(model.model) == 33 and list(head.f) == [2, 23, 27, 31]
        assert model.model[4].__class__.__name__ == layer4_name and isinstance(head, RAMPDetect)
        assert head.nl == 3 and head.p3_reactivation.use_ambiguity is ambiguity and head.p3_reactivation.use_channel is channel
        assert torch.allclose(head.stride.cpu(), torch.tensor([8.0, 16.0, 32.0]))

    for scale, (c_shallow, c_p3, detect_channels, repeats, dsc3k) in EXPECTED.items():
        model = YOLO(str(model_dir / f"yolov13{scale}-ramp.yaml")).model.to(device).eval()
        scpg, head = model.model[4], model.model[32]
        assert isinstance(scpg, SCPGDSC3k2) and isinstance(head, RAMPDetect)
        assert (scpg.c_shallow, scpg.c_p3) == (c_shallow, c_p3)
        assert len(scpg.m) == repeats and scpg.dsc3k_enabled is dsc3k
        assert head.c_shallow == c_shallow and [tower[0].conv.in_channels for tower in head.cv2] == detect_channels
        assert head.nl == 3 and torch.allclose(head.stride.cpu(), torch.tensor([8.0, 16.0, 32.0]))
        head.bias_init()
        if scale == "n":
            with torch.no_grad():
                prediction = model(torch.randn(1, 3, args.imgsz, args.imgsz, device=device))[0]
            assert torch.isfinite(prediction).all()
            model.fuse(verbose=False)
            with torch.no_grad():
                fused = model(torch.randn(1, 3, args.imgsz, args.imgsz, device=device))[0]
            assert torch.isfinite(fused).all()
    print("RAMP configuration verification passed for K1--K4 and n/s/l/x.")


if __name__ == "__main__":
    main()
