#!/usr/bin/env python3
"""Validate the exact CARM-style YOLO.load initialization used by SEAL workers."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

import ultralytics
from run_seal_ablation import STAGES


def assert_local_package() -> None:
    package_path = Path(ultralytics.__file__).resolve()
    if not package_path.is_relative_to(ROOT):
        raise RuntimeError(f"Wrong ultralytics package: {package_path}; expected under {ROOT}")


def load_stage(stage: str):
    from ultralytics import YOLO

    yaml_name, _ = STAGES[stage]
    model = YOLO(str(ROOT / "ultralytics/cfg/models/v13/seal_ablation" / yaml_name))
    model.load(str(ROOT / "yolov13n.pt"))
    if not model.ckpt:
        raise RuntimeError(f"{stage}: YOLO.load() did not preserve model.ckpt")
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda", action="store_true", help="also run the required amp=False CUDA train/eval/fuse smoke check")
    args = parser.parse_args()
    assert_local_package()
    pretrained = ROOT / "yolov13n.pt"
    if not pretrained.is_file():
        raise FileNotFoundError(pretrained)
    for stage in STAGES:
        load_stage(stage)
        print(f"PASS YOLO.load {stage}")
    if args.cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the requested SEAL runtime preflight")
        model = load_stage("s7").model.cuda().train()
        with torch.autocast(device_type="cuda", enabled=False):
            train_outputs = model(torch.zeros(1, 3, 128, 128, device="cuda"))
        if not all(torch.isfinite(output).all() for output in train_outputs):
            raise RuntimeError("non-finite amp=False CUDA training outputs")
        model.eval()
        with torch.no_grad(), torch.autocast(device_type="cuda", enabled=False):
            output = model(torch.zeros(1, 3, 128, 128, device="cuda"))
        prediction = output[0] if isinstance(output, tuple) else output
        if not torch.isfinite(prediction).all():
            raise RuntimeError("non-finite amp=False CUDA evaluation output")
        fused = model.fuse().eval()
        with torch.no_grad(), torch.autocast(device_type="cuda", enabled=False):
            fused_output = fused(torch.zeros(1, 3, 128, 128, device="cuda"))
        fused_prediction = fused_output[0] if isinstance(fused_output, tuple) else fused_output
        if not torch.isfinite(fused_prediction).all():
            raise RuntimeError("non-finite fused CUDA output")
        print("PASS CUDA amp=False train/eval/fuse")


if __name__ == "__main__":
    main()
