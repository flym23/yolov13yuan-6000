#!/usr/bin/env python3
"""Run static and CPU-module checks required before repaired DCRQ B3/B4 cleanup or rerun."""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

import yaml


EXPECTED = {
    "yolov13n-dcrq-b2.yaml": ("DSC3k2", "Concat"),
    "yolov13n-dcrq-b3.yaml": ("DAWRBDSC3k2", "Concat"),
    "yolov13n-dcrq-b4.yaml": ("DAWRBDSC3k2", "RCFConcat"),
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
    directory = root / "ultralytics" / "cfg" / "models" / "v13"
    for name, expected in EXPECTED.items():
        config = yaml.safe_load((directory / name).read_text(encoding="utf-8"))
        graph = list(config["backbone"]) + list(config["head"])
        assert len(graph) == 33, f"{name} must have exactly 33 layers"
        assert graph[4][2] == expected[0] and graph[20][2] == expected[1], f"{name} repaired anchors are wrong"
        assert list(graph[-1][0]) == [23, 27, 31] and float(config["quality_gain"]) == 0.25
    namespace = runpy.run_path(str(root / "tests" / "test_dcrq_repaired_modules.py"))
    namespace["main"]()
    print("DCRQ repaired YAML and module self-check passed")


if __name__ == "__main__":
    main()
