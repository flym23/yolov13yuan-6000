"""Canonical GSD-YOLOv13 factorial definitions and reused original-YOLOv13 references."""

from __future__ import annotations

from pathlib import Path


ALL_STAGES = ("g0", "g1", "g2", "g3", "g4", "g5", "g6", "g7")
TRAIN_STAGES = ALL_STAGES[1:]
MODEL_FILES = {
    "g0": "yolov13-gsd-t0-baseline.yaml",
    "g1": "yolov13-gsd-t1-scgp.yaml",
    "g2": "yolov13-gsd-t2-mcas.yaml",
    "g3": "yolov13-gsd-t3-cstd.yaml",
    "g4": "yolov13-gsd-t4-scgp-mcas.yaml",
    "g5": "yolov13-gsd-t5-scgp-cstd.yaml",
    "g6": "yolov13-gsd-t6-mcas-cstd.yaml",
    "g7": "yolov13-gsd-t7-full.yaml",
}
REFERENCE_SUMMARIES = {
    "l0": "/home/room305/ZZF/yolov13-6000/runs/test/lcer_dcra_20260722_045426_l0_baseline_summary.json",
    "p0": "/home/room305/ZZF/yolov13-6000/runs/test/spc_lcer_dcra_20260722_162019_p0_baseline_summary.json",
}
STRUCTURES = {
    "g0": {"innovations": {"SCGP": False, "MCAS": False, "CSTD": False}, "layer_4": "original DSC3k2", "layer_15": "nearest-neighbor P5→P4", "layer_32": "original Detect", "purpose": "reused original YOLOv13 baseline; no G0 training"},
    "g1": {"innovations": {"SCGP": True, "MCAS": False, "CSTD": False}, "layer_4": "SCPGDSC3k2 (SER + CPPG; H3 geometry)", "layer_15": "nearest-neighbor P5→P4", "layer_32": "original Detect", "purpose": "SCGP backbone only"},
    "g2": {"innovations": {"SCGP": False, "MCAS": True, "CSTD": False}, "layer_4": "original DSC3k2", "layer_15": "MCASUp (nearest/bilinear/low-pass bilinear convex semantic bases; 0.10 energy budget)", "layer_32": "original Detect", "purpose": "MCAS semantic-continuation neck only"},
    "g3": {"innovations": {"SCGP": False, "MCAS": False, "CSTD": True}, "layer_4": "original DSC3k2", "layer_15": "nearest-neighbor P5→P4", "layer_32": "CSTDDetect", "purpose": "cross-scale task-decoupled head only"},
    "g4": {"innovations": {"SCGP": True, "MCAS": True, "CSTD": False}, "layer_4": "SCPGDSC3k2 (SER + CPPG; H3 geometry)", "layer_15": "MCASUp", "layer_32": "original Detect", "purpose": "SCGP + MCAS"},
    "g5": {"innovations": {"SCGP": True, "MCAS": False, "CSTD": True}, "layer_4": "SCPGDSC3k2 (SER + CPPG; H3 geometry)", "layer_15": "nearest-neighbor P5→P4", "layer_32": "CSTDDetect", "purpose": "SCGP + CSTD; SBT-T5 structural counterpart, re-trained in GSD protocol"},
    "g6": {"innovations": {"SCGP": False, "MCAS": True, "CSTD": True}, "layer_4": "original DSC3k2", "layer_15": "MCASUp", "layer_32": "CSTDDetect", "purpose": "MCAS + CSTD"},
    "g7": {"innovations": {"SCGP": True, "MCAS": True, "CSTD": True}, "layer_4": "SCPGDSC3k2 (SER + CPPG; H3 geometry)", "layer_15": "MCASUp", "layer_32": "CSTDDetect", "purpose": "full GSD-YOLOv13"},
}


def resolve_model(root: Path, stage: str) -> Path:
    try:
        filename = MODEL_FILES[stage.lower()]
    except KeyError as error:
        raise ValueError(f"stage must be one of {', '.join(ALL_STAGES)}, got {stage!r}") from error
    return root / "ultralytics" / "cfg" / "models" / "v13" / filename
