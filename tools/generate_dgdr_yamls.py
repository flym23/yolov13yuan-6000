#!/usr/bin/env python3
"""Generate and structurally validate the reviewed DGDR-YOLOv13 YAML variants."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "ultralytics" / "cfg" / "models" / "v13"
BASE_PATH = MODEL_DIR / "yolov13.yaml"

DAPD = [-1, 1, "DAPD", [256, 3, 2, 1, 4, 0.10, 4]]
CAGB = [-1, 2, "CAGDSC3k2", [512, False, 0.25, 5, 2.0, 0.08, 4]]
SBRH = [[23, 27, 31], 1, "SBRHDetect", ["nc", 1.0, 12.0, 0.25, 0.10, True]]

VARIANTS = {
    "yolov13-dgdr-g1-dapd.yaml": (True, False, False),
    "yolov13-dgdr-g2-cagb.yaml": (False, True, False),
    "yolov13-dgdr-g3-sbrh.yaml": (False, False, True),
    "yolov13-dgdr-g4-dapd-cagb.yaml": (True, True, False),
    "yolov13-dgdr-g5-dapd-sbrh.yaml": (True, False, True),
    "yolov13-dgdr-g6-cagb-sbrh.yaml": (False, True, True),
    "yolov13-dgdr-g7-full.yaml": (True, True, True),
    "yolov13-dgdr.yaml": (True, True, True),
}


def build_variant(base: dict, use_dapd: bool, use_cagb: bool, use_sbrh: bool) -> dict:
    """Deep-copy original YOLOv13 and replace only reviewed layers 3, 4 and 32."""
    cfg = deepcopy(base)
    cfg["backbone"][3] = deepcopy(DAPD) if use_dapd else deepcopy(base["backbone"][3])
    cfg["backbone"][4] = deepcopy(CAGB) if use_cagb else deepcopy(base["backbone"][4])
    cfg["head"][-1] = deepcopy(SBRH) if use_sbrh else deepcopy(base["head"][-1])
    return cfg


def validate(cfg: dict, expected: tuple[bool, bool, bool]) -> None:
    """Reject index drift, class-count drift, or a misplaced DGDR replacement."""
    assert cfg["nc"] == 4, cfg["nc"]
    assert len(cfg["backbone"] + cfg["head"]) == 33
    assert cfg["head"][-1][0] == [23, 27, 31]
    expected_modules = (
        ("DAPD" if expected[0] else "Conv"),
        ("CAGDSC3k2" if expected[1] else "DSC3k2"),
        ("SBRHDetect" if expected[2] else "Detect"),
    )
    actual_modules = (cfg["backbone"][3][2], cfg["backbone"][4][2], cfg["head"][-1][2])
    assert actual_modules == expected_modules, (actual_modules, expected_modules)


def main() -> None:
    base = yaml.safe_load(BASE_PATH.read_text(encoding="utf-8"))
    assert base["nc"] == 4, "baseline yolov13.yaml must declare the URPC2020 four classes"
    for name, switches in VARIANTS.items():
        cfg = build_variant(base, *switches)
        validate(cfg, switches)
        target = MODEL_DIR / name
        target.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
        loaded = yaml.safe_load(target.read_text(encoding="utf-8"))
        validate(loaded, switches)
        print(target.relative_to(ROOT))


if __name__ == "__main__":
    main()
