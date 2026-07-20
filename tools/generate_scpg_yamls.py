#!/usr/bin/env python3
"""Generate SCPG-YOLOv13 H1--H5 YAMLs in the project's one-layer-per-line format."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "ultralytics" / "cfg" / "models" / "v13"
BASE = MODEL_DIR / "yolov13.yaml"

LAYER4 = "  - [-1, 2, DSC3k2,  [512, False, 0.25]]"
HEAD = "  - [[23, 27, 31], 1, Detect, [nc]] # Detect(P3, P4, P5)"
SCPG_TEMPLATE = "  - [[2, 3], 2, SCPGDSC3k2, [512, False, 0.25, 0.50, 5, 0.25, 1.50, 0.50, {detail:.2f}, {geom:.2f}, 4]]"
P3_HEAD = "  - [[23, 27, 31], 1, P3DecoupledDetect, [nc, 0.10, 4]] # Detect(P3, P4, P5)"

VARIANTS = {
    "h1-ser": (0.08, 0.00, False),
    "h2-cppg": (0.00, 0.08, False),
    "h3-ser-cppg": (0.08, 0.08, False),
    "h4-p3tda": (None, None, True),
    "h5-full": (0.08, 0.08, True),
    "scpg": (0.08, 0.08, True),
}


def validate(cfg: dict, key: str, detail: float | None, geom: float | None, p3_head: bool) -> None:
    assert cfg["nc"] == 4, cfg["nc"]
    assert len(cfg["backbone"] + cfg["head"]) == 33
    layer4, detect = cfg["backbone"][4], cfg["head"][-1]
    if key == "h4-p3tda":
        assert layer4[2] == "DSC3k2"
    else:
        assert layer4[0] == [2, 3] and layer4[2] == "SCPGDSC3k2"
        assert float(layer4[3][8]) == detail and float(layer4[3][9]) == geom
    assert detect[2] == ("P3DecoupledDetect" if p3_head else "Detect")
    assert detect[0] == [23, 27, 31]


def render(base_text: str, detail: float | None, geom: float | None, p3_head: bool) -> str:
    result = base_text
    if detail is not None:
        result = result.replace(LAYER4, SCPG_TEMPLATE.format(detail=detail, geom=geom), 1)
    if p3_head:
        result = result.replace(HEAD, P3_HEAD, 1)
    return result


def main() -> None:
    base_text = BASE.read_text(encoding="utf-8")
    for key, (detail, geom, p3_head) in VARIANTS.items():
        name = "yolov13-scpg.yaml" if key == "scpg" else f"yolov13-scpg-{key}.yaml"
        rendered = render(base_text, detail, geom, p3_head)
        target = MODEL_DIR / name
        target.write_text(rendered, encoding="utf-8")
        validate(yaml.safe_load(target.read_text(encoding="utf-8")), key, detail, geom, p3_head)
        print(target.relative_to(ROOT))


if __name__ == "__main__":
    main()
