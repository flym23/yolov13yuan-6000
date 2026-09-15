#!/usr/bin/env python3
"""Generate the BSRQ S0--S7 YAMLs from the verified DCRQ B1 parent YAML."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml


SOURCE_NAME = "yolov13n-dcrq-b1.yaml"
OUTPUT_NAMES = {f"S{index}": f"yolov13n-bsrq-s{index}.yaml" for index in range(8)}
EXPECTED_DETECT_INPUTS = [23, 27, 31]
STAGE_OPTIONS = {
    "S0": ("Conv", "FullPAD_Tunnel", "UDQDetect"),
    "S1": ("BoundedSpectralStem", "FullPAD_Tunnel", "UDQDetect"),
    "S2": ("Conv", "ReliabilityAdaptiveFullPAD", "UDQDetect"),
    "S3": ("Conv", "FullPAD_Tunnel", "SCQUDQDetect"),
    "S4": ("BoundedSpectralStem", "FullPAD_Tunnel", "SCQUDQDetect"),
    "S5": ("Conv", "ReliabilityAdaptiveFullPAD", "SCQUDQDetect"),
    "S6": ("BoundedSpectralStem", "ReliabilityAdaptiveFullPAD", "UDQDetect"),
    "S7": ("BoundedSpectralStem", "ReliabilityAdaptiveFullPAD", "SCQUDQDetect"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Absolute project root.")
    return parser.parse_args()


def validate_structure(config: dict) -> None:
    layers = list(config.get("backbone", [])) + list(config.get("head", []))
    if len(layers) != 33:
        raise ValueError(f"Expected the B1 parent to have 33 layers, found {len(layers)}.")
    if list(layers[32][0]) != EXPECTED_DETECT_INPUTS:
        raise ValueError(f"Expected Detect inputs {EXPECTED_DETECT_INPUTS}, found {layers[32][0]}.")
    if float(config.get("quality_gain", 0.0)) != 0.25:
        raise ValueError("The B1 parent must retain quality_gain=0.25.")


def validate_parent(config: dict) -> None:
    validate_structure(config)
    layers = list(config["backbone"]) + list(config["head"])
    if layers[0][2] != "Conv" or layers[23][2] != "FullPAD_Tunnel" or layers[32][2] != "UDQDetect":
        raise ValueError("The B1 parent does not have the expected Conv/FullPAD_Tunnel/UDQDetect anchors.")


def make_stage(parent: dict, stage: str) -> dict:
    stem, fullpad, head = STAGE_OPTIONS[stage]
    result = copy.deepcopy(parent)
    layers = result["backbone"] + result["head"]
    layers[0][2] = stem
    layers[0][3] = [64, 3, 2, 0.08, 8] if stem == "BoundedSpectralStem" else [64, 3, 2]
    layers[23][2] = fullpad
    layers[23][3] = [0.20, 8, True] if fullpad == "ReliabilityAdaptiveFullPAD" else []
    layers[32][2] = head
    if head == "SCQUDQDetect":
        layers[32][3] = [
            "nc",
            {
                "quality_mix": 0.50,
                "stat_strengths": [1.0, 0.5, 0.25],
                "detach_stats": True,
                "scq_max_deltas": [0.25, 0.15, 0.08],
                "scq_hidden": 8,
            },
        ]
    result["backbone"], result["head"] = layers[:9], layers[9:]
    result["bsrq_stage"] = stage
    result["bsrq_parent"] = SOURCE_NAME
    result["bsrq_structure"] = {"stem": stem, "p3_fullpad": fullpad, "detect": head}
    validate_structure(result)
    return result


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    if not root.is_absolute():
        raise ValueError("--root must be absolute")
    model_dir = root / "ultralytics" / "cfg" / "models" / "v13"
    source = model_dir / SOURCE_NAME
    if not source.is_file():
        raise FileNotFoundError(f"B1 parent YAML not found: {source}")
    parent = yaml.safe_load(source.read_text(encoding="utf-8"))
    validate_parent(parent)
    for stage, name in OUTPUT_NAMES.items():
        output = model_dir / name
        payload = make_stage(parent, stage)
        output.write_text(
            "# Generated from verified DCRQ B1; do not hand-edit.\n" + yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        print(f"generated {output}")


if __name__ == "__main__":
    main()
