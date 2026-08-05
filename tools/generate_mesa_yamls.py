#!/usr/bin/env python3
"""Generate MESA M0--M7 YAML configurations from the verified CARM-A3 topology."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "ultralytics/cfg/models/v13/carm_ablation/yolov13-carm-a3-macr.yaml"
OUTPUT = ROOT / "ultralytics/cfg/models/v13/mesa_ablation"
FLAGS = {
    "m1-hema": (True, False, False),
    "m2-gisr": (False, True, False),
    "m3-dmbr": (False, False, True),
    "m4-hema-gisr": (True, True, False),
    "m5-hema-dmbr": (True, False, True),
    "m6-gisr-dmbr": (False, True, True),
    "m7-full": (True, True, True),
}
MESA_ARGS = """[256, True, 0.50,
     0.08, 0.15, 0.25, 0.12, 0.30, 4, True,
     {hema}, {gisr}, {dmbr},
     0.04, 0.16, 0.70, 0.25,
     0.06, 0.20, 0.08, 0.35,
     0.25, 0.20, 0.80, 0.05, 0.12, 0.06]"""
LAYER21 = re.compile(r"(?ms)^  - \[\[2, 20, 17\], 2, MACRDSC3k2,\n     \[.*?\]\]\n(?=  - \[10, 1, Conv)")


def mesa_layer(flags: tuple[bool, bool, bool]) -> str:
    hema, gisr, dmbr = (str(value) for value in flags)
    return "  - [[2, 20, 17], 2, MESADSC3k2,\n     " + MESA_ARGS.format(hema=hema, gisr=gisr, dmbr=dmbr) + "]\n"


def main() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    if len(LAYER21.findall(source)) != 1:
        raise ValueError("expected exactly one CARM-A3 layer-21 declaration")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "yolov13-mesa-m0-a3.yaml").write_text(source, encoding="utf-8")
    for suffix, flags in FLAGS.items():
        generated = LAYER21.sub(mesa_layer(flags), source)
        (OUTPUT / f"yolov13-mesa-{suffix}.yaml").write_text(generated, encoding="utf-8")
    (ROOT / "ultralytics/cfg/models/v13/yolov13-mesa.yaml").write_text(
        (OUTPUT / "yolov13-mesa-m7-full.yaml").read_text(encoding="utf-8"), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
