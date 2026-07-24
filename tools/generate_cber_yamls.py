"""Generate CBER N3--N6 model YAMLs in the project's one-layer-per-line format."""

from __future__ import annotations

from pathlib import Path

import yaml

from cber_experiments import MODEL_FILES, STAGE_ORDER, STAGE_PARAMETERS


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "ultralytics" / "cfg" / "models" / "v13"
BASE = MODEL_DIR / "yolov13-cber.yaml"
LAYER4_PREFIX = "  - [[2, 3], 2, CBERSCPGDSC3k2, [512, False, 0.25, 0.50, 5, 0.25, 1.50, 0.50, 0.08, 0.08, 4,"


def render_layer4(params: dict[str, float]) -> str:
    return (
        "  - [[2, 3], 2, CBERSCPGDSC3k2, "
        "[512, False, 0.25, 0.50, 5, 0.25, 1.50, 0.50, 0.08, 0.08, 4, "
        f"{params['release_rho']:.2f}, {params['local_self']:.2f}, {params['co_weight']:.2f}]] # 4 CBER-SCPG"
    )


def main() -> None:
    base_text = BASE.read_text(encoding="utf-8")
    if LAYER4_PREFIX not in base_text:
        raise RuntimeError(f"CBER layer-4 template was not found in {BASE}")
    for stage in STAGE_ORDER:
        line = next(item for item in base_text.splitlines() if item.startswith(LAYER4_PREFIX))
        target = MODEL_DIR / MODEL_FILES[stage]
        target.write_text(base_text.replace(line, render_layer4(STAGE_PARAMETERS[stage]), 1), encoding="utf-8")
        cfg = yaml.safe_load(target.read_text(encoding="utf-8"))
        layer4 = cfg["backbone"][4]
        assert len(cfg["backbone"] + cfg["head"]) == 33
        assert layer4[0] == [2, 3] and layer4[2] == "CBERSCPGDSC3k2"
        assert [float(value) for value in layer4[3][11:14]] == list(STAGE_PARAMETERS[stage].values())
        print(target.relative_to(ROOT))


if __name__ == "__main__":
    main()
