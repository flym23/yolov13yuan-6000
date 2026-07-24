"""Canonical CBER-YOLOv13 N3--N6 ablation definitions and immutable reference metrics."""

from __future__ import annotations

from pathlib import Path


STAGE_ORDER = ("n3", "n4", "n5", "n6")
MODEL_FILES = {
    "n3": "yolov13-cber-n3-no-local.yaml",
    "n4": "yolov13-cber-n4-full.yaml",
    "n5": "yolov13-cber-n5-rho010.yaml",
    "n6": "yolov13-cber-n6-rho030.yaml",
}
STAGE_PARAMETERS = {
    "n3": {"release_rho": 0.20, "local_self": 1.00, "co_weight": 0.50},
    "n4": {"release_rho": 0.20, "local_self": 0.75, "co_weight": 0.50},
    "n5": {"release_rho": 0.10, "local_self": 0.75, "co_weight": 0.50},
    "n6": {"release_rho": 0.30, "local_self": 0.75, "co_weight": 0.50},
}
STRUCTURES = {
    stage: {
        "layer_3": "Conv",
        "layer_4": f"CBERSCPGDSC3k2(release_rho={params['release_rho']:.2f}, local_self={params['local_self']:.2f}, co_weight={params['co_weight']:.2f}; H3 geometry retained)",
        "layer_32": "Detect",
        "purpose": {
            "n3": "budgeted evidence routing without 3x3 local consensus",
            "n4": "full consensus-budgeted evidence routing",
            "n5": "test whether a tighter release budget suppresses useful detail",
            "n6": "test whether a wider release budget increases instability",
        }[stage],
    }
    for stage, params in STAGE_PARAMETERS.items()
}

# User-designated LCER-DCRA L0 global baseline, percent units, seed-aligned.
L0_PER_SEED = {0: 45.382590, 1: 45.395271, 2: 45.333289}
H3_PER_SEED = {0: 45.509644, 1: 45.519611, 2: 45.365285}
H3_MEAN = {"P": 81.935786, "R": 72.734677, "mAP50": 78.992631, "mAP75": 47.760937, "mAP50-95": 45.464847, "APS": 13.582038, "APM": 38.329134, "APL": 47.276530}
N4_MINIMUMS = {"P": 81.0, "R": 72.7, "mAP50": 79.0, "mAP75": 47.95, "mAP50-95": 45.65, "APS": 13.50, "APM": 38.45, "APL": 47.35}
N4_MAX_STD = 0.15
N4_MAX_GFLOPS = 7.40


def resolve_model(root: Path, stage: str) -> Path:
    try:
        filename = MODEL_FILES[stage.lower()]
    except KeyError as error:
        raise ValueError(f"stage must be one of {', '.join(STAGE_ORDER)}, got {stage!r}") from error
    return root / "ultralytics" / "cfg" / "models" / "v13" / filename
