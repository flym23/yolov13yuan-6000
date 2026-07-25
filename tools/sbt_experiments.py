"""Canonical SBT-YOLOv13 full-factorial definitions and reused baseline references."""

from __future__ import annotations

from pathlib import Path


ALL_STAGES = ("t0", "t1", "t2", "t3", "t4", "t5", "t6", "t7")
TRAIN_STAGES = ALL_STAGES[1:]
MODEL_FILES = {
    "t0": "yolov13-sbt-t0-baseline.yaml",
    "t1": "yolov13-sbt-t1-scgp.yaml",
    "t2": "yolov13-sbt-t2-bcra.yaml",
    "t3": "yolov13-sbt-t3-cstd.yaml",
    "t4": "yolov13-sbt-t4-scgp-bcra.yaml",
    "t5": "yolov13-sbt-t5-scgp-cstd.yaml",
    "t6": "yolov13-sbt-t6-bcra-cstd.yaml",
    "t7": "yolov13-sbt-t7-full.yaml",
}
REFERENCE_SUMMARIES = {
    "l0": "/home/room305/ZZF/yolov13-6000/runs/test/lcer_dcra_20260722_045426_l0_baseline_summary.json",
    "p0": "/home/room305/ZZF/yolov13-6000/runs/test/spc_lcer_dcra_20260722_162019_p0_baseline_summary.json",
}
STRUCTURES = {
    "t0": {
        "innovations": {"SCGP": False, "BCRA": False, "CSTD": False},
        "layer_4": "original DSC3k2",
        "layer_15": "nearest-neighbor P5→P4 upsample",
        "layer_32": "original Detect",
        "purpose": "reused original YOLOv13 reference; no new T0 training launched",
    },
    "t1": {
        "innovations": {"SCGP": True, "BCRA": False, "CSTD": False},
        "layer_4": "SCPGDSC3k2 (SER + CPPG; H3 geometry)",
        "layer_15": "nearest-neighbor P5→P4 upsample",
        "layer_32": "original Detect",
        "purpose": "Innovation 1: shallow geometry preservation only",
    },
    "t2": {
        "innovations": {"SCGP": False, "BCRA": True, "CSTD": False},
        "layer_4": "original DSC3k2",
        "layer_15": "BCRAUp (3×3 candidates, boundary query, entropy confidence, 0.20 energy budget)",
        "layer_32": "original Detect",
        "purpose": "Innovation 2: boundary-conditioned P5→P4 reassembly only",
    },
    "t3": {
        "innovations": {"SCGP": False, "BCRA": False, "CSTD": True},
        "layer_4": "original DSC3k2",
        "layer_15": "nearest-neighbor P5→P4 upsample",
        "layer_32": "CSTDDetect (deep semantic classification / shallow boundary regression context)",
        "purpose": "Innovation 3: cross-scale task-decoupled head only",
    },
    "t4": {
        "innovations": {"SCGP": True, "BCRA": True, "CSTD": False},
        "layer_4": "SCPGDSC3k2 (SER + CPPG; H3 geometry)",
        "layer_15": "BCRAUp (3×3 candidates, boundary query, entropy confidence, 0.20 energy budget)",
        "layer_32": "original Detect",
        "purpose": "SCGP + BCRA complementarity",
    },
    "t5": {
        "innovations": {"SCGP": True, "BCRA": False, "CSTD": True},
        "layer_4": "SCPGDSC3k2 (SER + CPPG; H3 geometry)",
        "layer_15": "nearest-neighbor P5→P4 upsample",
        "layer_32": "CSTDDetect (deep semantic classification / shallow boundary regression context)",
        "purpose": "SCGP + CSTD complementarity",
    },
    "t6": {
        "innovations": {"SCGP": False, "BCRA": True, "CSTD": True},
        "layer_4": "original DSC3k2",
        "layer_15": "BCRAUp (3×3 candidates, boundary query, entropy confidence, 0.20 energy budget)",
        "layer_32": "CSTDDetect (deep semantic classification / shallow boundary regression context)",
        "purpose": "BCRA + CSTD complementarity",
    },
    "t7": {
        "innovations": {"SCGP": True, "BCRA": True, "CSTD": True},
        "layer_4": "SCPGDSC3k2 (SER + CPPG; H3 geometry)",
        "layer_15": "BCRAUp (3×3 candidates, boundary query, entropy confidence, 0.20 energy budget)",
        "layer_32": "CSTDDetect (deep semantic classification / shallow boundary regression context)",
        "purpose": "full SBT-YOLOv13",
    },
}


def resolve_model(root: Path, stage: str) -> Path:
    try:
        filename = MODEL_FILES[stage.lower()]
    except KeyError as error:
        raise ValueError(f"stage must be one of {', '.join(ALL_STAGES)}, got {stage!r}") from error
    return root / "ultralytics" / "cfg" / "models" / "v13" / filename
