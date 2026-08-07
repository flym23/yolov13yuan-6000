"""Static and CPU forward validation for every SEAL ablation configuration."""

from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

import ultralytics
from ultralytics.nn.tasks import DetectionModel, yaml_model_load


EXPECTED_STAGES = {
    "s0-carm-a3": (False, False, False),
    "s1-dsrf": (True, False, False),
    "s2-sarb": (False, True, False),
    "s3-qad": (False, False, True),
    "s4-dsrf-sarb": (True, True, False),
    "s5-dsrf-qad": (True, False, True),
    "s6-sarb-qad": (False, True, True),
    "s7-full": (True, True, True),
}


def assert_local_ultralytics() -> None:
    """Prevent accidental imports from the system Ultralytics package on remote workers."""
    package_path = Path(ultralytics.__file__).resolve()
    if not package_path.is_relative_to(ROOT):
        raise RuntimeError(f"Wrong ultralytics package: {package_path}; expected under {ROOT}")


def build(path: Path, scale: str | None = None) -> DetectionModel:
    spec = yaml_model_load(path)
    if scale:
        spec = deepcopy(spec)
        spec["scale"] = scale
    return DetectionModel(cfg=spec, ch=3, nc=None, verbose=False)


def check_stage(stage: str, path: Path) -> None:
    dsrf, sarb, qad = EXPECTED_STAGES[stage]
    model = build(path)
    layers = model.model
    if len(layers) != 33:
        raise AssertionError(f"{stage}: expected 33 layers, got {len(layers)}")
    expected_names = ("DSRFDSC3k2" if dsrf else "DSC3k2", "SARBDSC3k2" if sarb else "DSC3k2")
    actual_names = (layers[2].__class__.__name__, layers[17].__class__.__name__)
    if actual_names != expected_names:
        raise AssertionError(f"{stage}: layers 2/17 are {actual_names}, expected {expected_names}")
    expected_head = "QualityAlignedDecoupledDetect" if qad else "Detect"
    if layers[32].__class__.__name__ != expected_head:
        raise AssertionError(f"{stage}: detect head is {layers[32].__class__.__name__}, expected {expected_head}")
    if tuple(model.stride.tolist()) != (8.0, 16.0, 32.0):
        raise AssertionError(f"{stage}: invalid stride {model.stride.tolist()}")
    with torch.no_grad():
        model.eval()
        prediction = model(torch.zeros(1, 3, 128, 128))
    if not torch.isfinite(prediction[0] if isinstance(prediction, tuple) else prediction).all():
        raise AssertionError(f"{stage}: non-finite CPU prediction")


def check_scales(full_path: Path) -> None:
    for scale in "nslx":
        model = build(full_path, scale)
        if tuple(model.stride.tolist()) != (8.0, 16.0, 32.0):
            raise AssertionError(f"scale={scale}: invalid stride {model.stride.tolist()}")
        dsrf = model.model[2]
        sarb = model.model[17]
        if dsrf.__class__.__name__ != "DSRFDSC3k2" or sarb.__class__.__name__ != "SARBDSC3k2":
            raise AssertionError(f"scale={scale}: SEAL replacement missing")
        expected_dsrf_dsc3k = scale in "lx"
        # SARB preserves CARM-A3's P4 DSC3k2 setting from YAML (True at every scale);
        # the parser's L/X override remains explicit for backward compatibility.
        expected_sarb_dsc3k = True
        if dsrf.dsc3k_enabled != expected_dsrf_dsc3k or sarb.dsc3k_enabled != expected_sarb_dsc3k:
            raise AssertionError(
                f"scale={scale}: dsc3k parsing is incorrect "
                f"(DSRF={dsrf.dsc3k_enabled}, SARB={sarb.dsc3k_enabled}, "
                f"expected=({expected_dsrf_dsc3k}, {expected_sarb_dsc3k}))"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", type=Path, default=ROOT / "ultralytics/cfg/models/v13/seal_ablation")
    args = parser.parse_args()
    assert_local_ultralytics()
    for stage in EXPECTED_STAGES:
        path = args.config_dir / f"yolov13-seal-{stage}.yaml"
        if not path.is_file():
            raise FileNotFoundError(path)
        check_stage(stage, path)
        print(f"PASS {stage}")
    check_scales(args.config_dir / "yolov13-seal-s7-full.yaml")
    print("PASS scales=n/s/l/x")


if __name__ == "__main__":
    main()
