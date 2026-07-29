"""Canonical GMR-YOLOv13 factorial definitions and reused original-YOLOv13 references."""

from __future__ import annotations

from pathlib import Path


ALL_STAGES = ("r0", "r1", "r2", "r3", "r4", "r5", "r6", "r7")
TRAIN_STAGES = ALL_STAGES[1:]
# Every formal ablation uses exactly three independent seeds.  This is both the
# statistical protocol and the hard upper bound for concurrent training jobs.
SEEDS = (0, 1, 2)
MODEL_FILES = {
    "r0": "yolov13-gmr-r0-baseline.yaml",
    "r1": "yolov13-gmr-r1-scgp.yaml",
    "r2": "yolov13-gmr-r2-mcas.yaml",
    "r3": "yolov13-gmr-r3-gimr.yaml",
    "r4": "yolov13-gmr-r4-scgp-mcas.yaml",
    "r5": "yolov13-gmr-r5-scgp-gimr.yaml",
    "r6": "yolov13-gmr-r6-mcas-gimr.yaml",
    "r7": "yolov13-gmr-r7-full.yaml",
}
REFERENCE_SUMMARIES = {
    "l0": "/home/room305/ZZF/yolov13-6000/runs/test/lcer_dcra_20260722_045426_l0_baseline_summary.json",
    "p0": "/home/room305/ZZF/yolov13-6000/runs/test/spc_lcer_dcra_20260722_162019_p0_baseline_summary.json",
}
STRUCTURES = {
    "r0": {
        "innovations": {"SCGP": False, "MCAS": False, "GIMR": False},
        "layer_4": "original DSC3k2",
        "layer_15": "nearest-neighbor P5-to-P4",
        "layer_32": "original Detect",
        "purpose": "reused L0/P0 original-YOLOv13 references; no R0 training",
    },
    "r1": {
        "innovations": {"SCGP": True, "MCAS": False, "GIMR": False},
        "layer_4": "SCPGDSC3k2 (SER + CPPG; H3 geometry)",
        "layer_15": "nearest-neighbor P5-to-P4",
        "layer_32": "original Detect",
        "purpose": "SCGP backbone only",
    },
    "r2": {
        "innovations": {"SCGP": False, "MCAS": True, "GIMR": False},
        "layer_4": "original DSC3k2",
        "layer_15": "MCASUp (three semantic bases; 0.10 energy budget)",
        "layer_32": "original Detect",
        "purpose": "MCAS neck only",
    },
    "r3": {
        "innovations": {"SCGP": False, "MCAS": False, "GIMR": True},
        "layer_4": "original DSC3k2",
        "layer_15": "nearest-neighbor P5-to-P4",
        "layer_32": "GIMRDetect (P3 classification only; detached P3/P4 evidence; 0.04 budget)",
        "purpose": "GIMR micro-target reconciliation only",
    },
    "r4": {
        "innovations": {"SCGP": True, "MCAS": True, "GIMR": False},
        "layer_4": "SCPGDSC3k2 (SER + CPPG; H3 geometry)",
        "layer_15": "MCASUp (three semantic bases; 0.10 energy budget)",
        "layer_32": "original Detect",
        "purpose": "SCGP + MCAS historical G4 structure re-run under GMR protocol",
    },
    "r5": {
        "innovations": {"SCGP": True, "MCAS": False, "GIMR": True},
        "layer_4": "SCPGDSC3k2 (SER + CPPG; H3 geometry)",
        "layer_15": "nearest-neighbor P5-to-P4",
        "layer_32": "GIMRDetect (P3 classification only; detached P3/P4 evidence; 0.04 budget)",
        "purpose": "SCGP + GIMR",
    },
    "r6": {
        "innovations": {"SCGP": False, "MCAS": True, "GIMR": True},
        "layer_4": "original DSC3k2",
        "layer_15": "MCASUp (three semantic bases; 0.10 energy budget)",
        "layer_32": "GIMRDetect (P3 classification only; detached P3/P4 evidence; 0.04 budget)",
        "purpose": "MCAS + GIMR",
    },
    "r7": {
        "innovations": {"SCGP": True, "MCAS": True, "GIMR": True},
        "layer_4": "SCPGDSC3k2 (SER + CPPG; H3 geometry)",
        "layer_15": "MCASUp (three semantic bases; 0.10 energy budget)",
        "layer_32": "GIMRDetect (P3 classification only; detached P3/P4 evidence; 0.04 budget)",
        "purpose": "full GMR-YOLOv13",
    },
}


def resolve_model(root: Path, stage: str) -> Path:
    try:
        filename = MODEL_FILES[stage.lower()]
    except KeyError as error:
        raise ValueError(f"stage must be one of {', '.join(ALL_STAGES)}, got {stage!r}") from error
    return root / "ultralytics" / "cfg" / "models" / "v13" / filename
