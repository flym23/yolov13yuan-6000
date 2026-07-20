"""Canonical SCPG-YOLOv13 H1--H5 ablation definitions."""

from __future__ import annotations

from pathlib import Path

STAGE_ORDER = ("h1", "h2", "h3", "h4", "h5")
MODEL_FILES = {
    "h1": "yolov13-scpg-h1-ser.yaml",
    "h2": "yolov13-scpg-h2-cppg.yaml",
    "h3": "yolov13-scpg-h3-ser-cppg.yaml",
    "h4": "yolov13-scpg-h4-p3tda.yaml",
    "h5": "yolov13-scpg-h5-full.yaml",
}
STRUCTURES = {
    "h1": {
        "layer_3": "Conv",
        "layer_4": "SCPGDSC3k2(detail_gain=0.08, geom_gain=0.00)",
        "layer_32": "Detect",
    },
    "h2": {
        "layer_3": "Conv",
        "layer_4": "SCPGDSC3k2(detail_gain=0.00, geom_gain=0.08)",
        "layer_32": "Detect",
    },
    "h3": {
        "layer_3": "Conv",
        "layer_4": "SCPGDSC3k2(detail_gain=0.08, geom_gain=0.08)",
        "layer_32": "Detect",
    },
    "h4": {
        "layer_3": "Conv",
        "layer_4": "DSC3k2",
        "layer_32": "P3DecoupledDetect(max_gain=0.10, reduction=4)",
    },
    "h5": {
        "layer_3": "Conv",
        "layer_4": "SCPGDSC3k2(detail_gain=0.08, geom_gain=0.08)",
        "layer_32": "P3DecoupledDetect(max_gain=0.10, reduction=4)",
    },
}


def resolve_model(root: Path, stage: str) -> Path:
    try:
        filename = MODEL_FILES[stage.lower()]
    except KeyError as error:
        raise ValueError(f"stage must be one of {', '.join(STAGE_ORDER)}, got {stage!r}") from error
    return root / "ultralytics" / "cfg" / "models" / "v13" / filename

