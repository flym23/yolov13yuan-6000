#!/usr/bin/env python3
"""Build and smoke-test all reviewed DGDR configurations before training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ultralytics import YOLO  # noqa: E402
from ultralytics.nn.modules import CAGDSC3k2, DAPD, SBRHDetect  # noqa: E402


EXPECTED = {
    "n": ((64, 64), (64, 128), [64, 128, 256], 1, False),
    "s": ((128, 128), (128, 256), [128, 256, 512], 1, False),
    "l": ((256, 256), (256, 512), [256, 512, 512], 2, True),
    "x": ((384, 384), (384, 512), [384, 512, 512], 2, True),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT_DIR)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", type=int, default=640)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = args.root.resolve() / "ultralytics" / "cfg" / "models" / "v13"
    device = torch.device(args.device)
    for suffix in ("g1-dapd", "g2-cagb", "g3-sbrh", "g7-full"):
        model = YOLO(str(model_dir / f"yolov13n-dgdr-{suffix}.yaml")).model
        assert len(model.model) == 33 and list(model.model[32].f) == [23, 27, 31]
    for scale, (dapd_channels, cagb_channels, detect_channels, repeats, dsc3k) in EXPECTED.items():
        model = YOLO(str(model_dir / f"yolov13{scale}-dgdr.yaml")).model.to(device).eval()
        layers = model.model
        assert isinstance(layers[3], DAPD) and isinstance(layers[4], CAGDSC3k2) and isinstance(layers[32], SBRHDetect)
        assert (layers[3].conv.in_channels, layers[3].conv.out_channels) == dapd_channels
        assert (layers[4].cv1.conv.in_channels, layers[4].cv2.conv.out_channels) == cagb_channels
        assert len(layers[4].m) == repeats and layers[4].dsc3k_enabled is dsc3k
        assert [tower[0].conv.in_channels for tower in layers[32].cv2] == detect_channels
        assert torch.allclose(layers[32].stride.cpu(), torch.tensor([8.0, 16.0, 32.0]))
        if scale == "n":
            with torch.no_grad():
                prediction = model(torch.randn(1, 3, args.imgsz, args.imgsz, device=device))[0]
            assert torch.isfinite(prediction).all()
            model.fuse(verbose=False)
            with torch.no_grad():
                fused = model(torch.randn(1, 3, args.imgsz, args.imgsz, device=device))[0]
            assert torch.isfinite(fused).all()
    print("DGDR configuration verification passed for n/s/l/x.")


if __name__ == "__main__":
    main()
