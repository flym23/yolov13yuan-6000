"""Canonical CPCR-YOLOv13 L1--L5 ablation definitions and L4 gate criteria."""

from __future__ import annotations

from pathlib import Path


STAGE_ORDER = ("l1", "l2", "l3", "l4")
CONDITIONAL_STAGE = "l5"
ALL_STAGES = (*STAGE_ORDER, CONDITIONAL_STAGE)
MODEL_FILES = {
    "l1": "yolov13-cpcr-l1-prototype.yaml",
    "l2": "yolov13-cpcr-l2-spatial.yaml",
    "l3": "yolov13-cpcr-l3-class.yaml",
    "l4": "yolov13-cpcr-l4-full.yaml",
    "l5": "yolov13-cpcr-l5-h3-full.yaml",
}
STRUCTURES = {
    "l1": {"backbone": "original DSC3k2", "head": "CPCRDetect: prototype candidate only", "purpose": "detached class-prototype candidate baseline"},
    "l2": {"backbone": "original DSC3k2", "head": "CPCRDetect: spatial prior + local support", "purpose": "isolate shallow spatial evidence"},
    "l3": {"backbone": "original DSC3k2", "head": "CPCRDetect: spatial prior + class gate + local support", "purpose": "test class-specific recovery"},
    "l4": {"backbone": "original DSC3k2", "head": "CPCRDetect: spatial prior + class gate + local support + DFL localization guard", "purpose": "full primary CPCR model"},
    "l5": {"backbone": "SCPG-H3 at layer 4", "head": "full CPCRDetect", "purpose": "conditional H3 interaction; launch only after L4 success"},
}
BASELINE_PER_SEED = {0: {"R": 73.805, "mAP50-95": 45.108}, 1: {"R": 73.976, "mAP50-95": 45.696}, 2: {"R": 73.157, "mAP50-95": 45.188}}
L4_THRESHOLDS = {"P": 79.8, "R": 73.7, "mAP50": 79.1, "mAP75": 47.7, "mAP50-95": 45.65, "APS": 13.7, "APM": 38.4, "APL": 47.0}
K1_REFERENCE = {"mean_mAP50-95": 45.450, "best_mAP50-95": 45.817}


def resolve_model(root: Path, stage: str) -> Path:
    try:
        filename = MODEL_FILES[stage.lower()]
    except KeyError as error:
        raise ValueError(f"stage must be one of {', '.join(ALL_STAGES)}, got {stage!r}") from error
    return root / "ultralytics" / "cfg" / "models" / "v13" / filename
