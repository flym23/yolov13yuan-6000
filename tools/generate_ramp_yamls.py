#!/usr/bin/env python3
"""Generate one-layer-per-line RAMP-YOLOv13 K1--K4 YAMLs."""

from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "ultralytics" / "cfg" / "models" / "v13"
BASE = MODEL_DIR / "yolov13.yaml"
LAYER4 = "  - [-1, 2, DSC3k2,  [512, False, 0.25]]"
HEAD = "  - [[23, 27, 31], 1, Detect, [nc]] # Detect(P3, P4, P5)"
SCPG_LAYER4 = "  - [[2, 3], 2, SCPGDSC3k2, [512, False, 0.25, 0.50, 5, 0.25, 1.50, 0.50, 0.08, 0.08, 4]]"
RAMP_HEAD = "  - [[2, 23, 27, 31], 1, RAMPDetect, [nc, 0.10, 4, -4.0, {ambiguity}, {channel}]] # P2 guidance + Detect(P3, P4, P5)"

VARIANTS = {
    "k1-baseline": (False, True, True),
    "k2-prior": (True, False, False),
    "k3-ambiguity": (True, True, False),
    "k4-full": (True, True, True),
    "ramp": (True, True, True),
}


def render(base_text: str, use_scpg: bool, use_ambiguity: bool, use_channel: bool) -> str:
    rendered = base_text
    if use_scpg:
        rendered = rendered.replace(LAYER4, SCPG_LAYER4, 1)
    return rendered.replace(HEAD, RAMP_HEAD.format(ambiguity=str(use_ambiguity), channel=str(use_channel)), 1)


def validate(config: dict, key: str, use_scpg: bool, use_ambiguity: bool, use_channel: bool) -> None:
    layers = config["backbone"] + config["head"]
    assert config["nc"] == 4 and len(layers) == 33
    layer4, head = config["backbone"][4], config["head"][-1]
    assert layer4[2] == ("SCPGDSC3k2" if use_scpg else "DSC3k2")
    assert head[0] == [2, 23, 27, 31] and head[2] == "RAMPDetect"
    assert bool(head[3][4]) is use_ambiguity and bool(head[3][5]) is use_channel, key


def main() -> None:
    base_text = BASE.read_text(encoding="utf-8")
    for key, variant in VARIANTS.items():
        use_scpg, use_ambiguity, use_channel = variant
        target_name = "yolov13-ramp.yaml" if key == "ramp" else f"yolov13-ramp-{key}.yaml"
        target = MODEL_DIR / target_name
        target.write_text(render(base_text, *variant), encoding="utf-8")
        validate(yaml.safe_load(target.read_text(encoding="utf-8")), key, *variant)
        print(target.relative_to(ROOT))


if __name__ == "__main__":
    main()
