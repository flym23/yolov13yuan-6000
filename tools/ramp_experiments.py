"""Canonical RAMP-YOLOv13 K1--K4 ablation definitions."""

from __future__ import annotations

from pathlib import Path


STAGE_ORDER = ("k1", "k2", "k3", "k4")
MODEL_FILES = {
    "k1": "yolov13-ramp-k1-baseline.yaml",
    "k2": "yolov13-ramp-k2-prior.yaml",
    "k3": "yolov13-ramp-k3-ambiguity.yaml",
    "k4": "yolov13-ramp-k4-full.yaml",
}
STRUCTURES = {
    "k0": {
        "backbone": "Existing SCPG H3: SCPGDSC3k2(detail_gain=0.08, geom_gain=0.08)",
        "head": "Original Detect(P3, P4, P5)",
        "purpose": "Existing stable SCPG reference; not retrained.",
    },
    "k1": {
        "backbone": "Original DSC3k2 at layer 4",
        "head": "RAMPDetect(P2 guidance; ambiguity=True; channel=True)",
        "purpose": "Test whether RAMP independently restores recall from the baseline backbone.",
    },
    "k2": {
        "backbone": "SCPG H3: SER + CPPG at layer 4",
        "head": "RAMPDetect(P2 prior only; ambiguity=False; channel=False)",
        "purpose": "Isolate spatial shallow-evidence reactivation.",
    },
    "k3": {
        "backbone": "SCPG H3: SER + CPPG at layer 4",
        "head": "RAMPDetect(P2 prior + ambiguity; channel=False)",
        "purpose": "Measure ambiguity selection without channel multiplexing.",
    },
    "k4": {
        "backbone": "SCPG H3: SER + CPPG at layer 4",
        "head": "RAMPDetect(P2 prior + ambiguity + channel multiplexing)",
        "purpose": "Full RAMP-YOLOv13.",
    },
}


def resolve_model(root: Path, stage: str) -> Path:
    try:
        filename = MODEL_FILES[stage.lower()]
    except KeyError as error:
        raise ValueError(f"stage must be one of {', '.join(STAGE_ORDER)}, got {stage!r}") from error
    return root / "ultralytics" / "cfg" / "models" / "v13" / filename
