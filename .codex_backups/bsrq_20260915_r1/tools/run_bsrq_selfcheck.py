#!/usr/bin/env python3
"""Run dependency-light BSRQ source checks before GPU model construction or training."""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

import yaml


EXPECTED = {
    "S0": ("Conv", "FullPAD_Tunnel", "UDQDetect"),
    "S1": ("BoundedSpectralStem", "FullPAD_Tunnel", "UDQDetect"),
    "S2": ("Conv", "ReliabilityAdaptiveFullPAD", "UDQDetect"),
    "S3": ("Conv", "FullPAD_Tunnel", "SCQUDQDetect"),
    "S4": ("BoundedSpectralStem", "FullPAD_Tunnel", "SCQUDQDetect"),
    "S5": ("Conv", "ReliabilityAdaptiveFullPAD", "SCQUDQDetect"),
    "S6": ("BoundedSpectralStem", "ReliabilityAdaptiveFullPAD", "UDQDetect"),
    "S7": ("BoundedSpectralStem", "ReliabilityAdaptiveFullPAD", "SCQUDQDetect"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_absolute():
        raise ValueError("--root must be absolute")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    model_dir = root / "ultralytics" / "cfg" / "models" / "v13"
    for stage, expected in EXPECTED.items():
        path = model_dir / f"yolov13n-bsrq-{stage.lower()}.yaml"
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        layers = list(config["backbone"]) + list(config["head"])
        assert len(layers) == 33, f"{stage} must remain a 33-layer graph"
        assert (layers[0][2], layers[23][2], layers[32][2]) == expected, f"{stage} structure is incorrect"
        assert list(layers[32][0]) == [23, 27, 31], f"{stage} detect inputs changed"
        assert float(config["quality_gain"]) == 0.25, f"{stage} quality_gain changed"
    namespace = runpy.run_path(str(root / "tests" / "test_bsrq_modules.py"))
    namespace["main"]()
    print("BSRQ YAML and module self-check passed")


if __name__ == "__main__":
    main()
