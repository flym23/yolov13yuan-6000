"""Canonical baseline and reviewed DGDR-YOLOv13 ablation definitions."""

from __future__ import annotations

from pathlib import Path


STAGE_ORDER = ("baseline", "g1", "g2", "g3", "g4", "g5", "g6", "g7")
MODEL_FILES = {
    "baseline": "yolov13.yaml",
    "g1": "yolov13-dgdr-g1-dapd.yaml",
    "g2": "yolov13-dgdr-g2-cagb.yaml",
    "g3": "yolov13-dgdr-g3-sbrh.yaml",
    "g4": "yolov13-dgdr-g4-dapd-cagb.yaml",
    "g5": "yolov13-dgdr-g5-dapd-sbrh.yaml",
    "g6": "yolov13-dgdr-g6-cagb-sbrh.yaml",
    "g7": "yolov13-dgdr-g7-full.yaml",
}
STRUCTURES = {
    "baseline": {"layer_3": "Conv", "layer_4": "DSC3k2", "layer_32": "Detect"},
    "g1": {"layer_3": "DAPD", "layer_4": "DSC3k2", "layer_32": "Detect"},
    "g2": {"layer_3": "Conv", "layer_4": "CAGDSC3k2", "layer_32": "Detect"},
    "g3": {"layer_3": "Conv", "layer_4": "DSC3k2", "layer_32": "SBRHDetect"},
    "g4": {"layer_3": "DAPD", "layer_4": "CAGDSC3k2", "layer_32": "Detect"},
    "g5": {"layer_3": "DAPD", "layer_4": "DSC3k2", "layer_32": "SBRHDetect"},
    "g6": {"layer_3": "Conv", "layer_4": "CAGDSC3k2", "layer_32": "SBRHDetect"},
    "g7": {"layer_3": "DAPD", "layer_4": "CAGDSC3k2", "layer_32": "SBRHDetect"},
}


def resolve_model(root: Path, stage: str) -> Path:
    """Return a reviewed stage's physical YAML and reject unknown stages."""
    try:
        filename = MODEL_FILES[stage.lower()]
    except KeyError as error:
        raise ValueError(f"stage must be one of {', '.join(STAGE_ORDER)}, got {stage!r}") from error
    return root / "ultralytics" / "cfg" / "models" / "v13" / filename
