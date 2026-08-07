"""Generate the eight SEAL ablation YAML files from the frozen CARM-A3 control."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "ultralytics/cfg/models/v13/carm_ablation/yolov13-carm-a3-macr.yaml"
TARGET_DIR = ROOT / "ultralytics/cfg/models/v13/seal_ablation"
MAIN_TARGET = ROOT / "ultralytics/cfg/models/v13/yolov13-seal.yaml"

DSRF = """- [-1, 2, DSRFDSC3k2,
     [256, False, 0.25, 0.08, 0.15, 0.25, 0.35, 0.10, 0.20, 4, True]]"""
SARB = """- [[16, 9], 2, SARBDSC3k2,
     [512, True, 0.50, 0.06, 0.30, 0.25, 0.06, 4, True, True]]"""
QAD = """- [[23, 27, 31], 1, QualityAlignedDecoupledDetect,
     [nc, 0.06, 0.05, 0.06, 0.05, 4, True, True]]"""

REPLACEMENTS = {
    "dsrf": ("- [-1, 2, DSC3k2, [256, False, 0.25]]", DSRF),
    "sarb": ("- [-1, 2, DSC3k2, [512, True]]", SARB),
    "qad": ("- [[23, 27, 31], 1, Detect, [nc]]", QAD),
}
STAGES = {
    "s0-carm-a3": (),
    "s1-dsrf": ("dsrf",),
    "s2-sarb": ("sarb",),
    "s3-qad": ("qad",),
    "s4-dsrf-sarb": ("dsrf", "sarb"),
    "s5-dsrf-qad": ("dsrf", "qad"),
    "s6-sarb-qad": ("sarb", "qad"),
    "s7-full": ("dsrf", "sarb", "qad"),
}


def render(stage: str, modules: tuple[str, ...]) -> str:
    """Return one complete model definition with deterministic, validated substitutions."""
    text = SOURCE.read_text(encoding="utf-8")
    for module in modules:
        old, new = REPLACEMENTS[module]
        if text.count(old) < 1:
            raise RuntimeError(f"{stage}: substitution target for {module} is missing")
        text = text.replace(old, new, 1)
    return f"# SEAL-YOLOv13 {stage}; generated from frozen CARM-A3.\n{text}"


def main() -> None:
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    for stage, modules in STAGES.items():
        (TARGET_DIR / f"yolov13-seal-{stage}.yaml").write_text(render(stage, modules), encoding="utf-8")
    MAIN_TARGET.write_text(render("s7-full", STAGES["s7-full"]), encoding="utf-8")
    print(f"generated {len(STAGES)} ablations and {MAIN_TARGET.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
