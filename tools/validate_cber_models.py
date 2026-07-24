"""Build and verify CBER model variants before long-running training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from cber_experiments import MODEL_FILES, STAGE_ORDER, STAGE_PARAMETERS  # noqa: E402
from ultralytics import YOLO  # noqa: E402
from ultralytics.nn.modules import CBERSCPGDSC3k2  # noqa: E402


EXPECTED = {
    "n": (64, 64, 128, 1, False),
    "s": (128, 128, 256, 1, False),
    "l": (256, 256, 512, 2, True),
    "x": (384, 384, 512, 2, True),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT_DIR)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root, device = args.root.resolve(), torch.device(args.device)
    model_dir = root / "ultralytics" / "cfg" / "models" / "v13"
    for stage in STAGE_ORDER:
        model = YOLO(str(model_dir / MODEL_FILES[stage])).model
        layer4, detect = model.model[4], model.model[32]
        params = STAGE_PARAMETERS[stage]
        assert len(model.model) == 33 and list(detect.f) == [23, 27, 31]
        assert isinstance(layer4, CBERSCPGDSC3k2)
        assert layer4.detail_router.release_rho == params["release_rho"]
        assert layer4.detail_router.local_self == params["local_self"]
        assert layer4.detail_router.co_weight == params["co_weight"]
        assert torch.allclose(detect.stride.cpu(), torch.tensor([8.0, 16.0, 32.0]))

    for scale, (c_shallow, c_p3, c2, repeats, dsc3k) in EXPECTED.items():
        model = YOLO(str(model_dir / f"yolov13{scale}-cber-n4-full.yaml")).model.to(device).eval()
        layer4, detect = model.model[4], model.model[32]
        assert isinstance(layer4, CBERSCPGDSC3k2)
        assert (layer4.c_shallow, layer4.c_p3, layer4.c2) == (c_shallow, c_p3, c2)
        assert len(layer4.m) == repeats and layer4.dsc3k_enabled is dsc3k
        assert list(detect.f) == [23, 27, 31]
        assert torch.allclose(detect.stride.cpu(), torch.tensor([8.0, 16.0, 32.0]))
        if scale == "n":
            with torch.no_grad():
                prediction = model(torch.randn(1, 3, args.imgsz, args.imgsz, device=device))[0]
            assert torch.isfinite(prediction).all()
            model.fuse(verbose=False)
            with torch.no_grad():
                fused = model(torch.randn(1, 3, args.imgsz, args.imgsz, device=device))[0]
            assert torch.isfinite(fused).all()
    print("CBER configuration verification passed for N3--N6 and n/s/l/x scales.")


if __name__ == "__main__":
    main()
