#!/usr/bin/env python3
"""GPU preflight for all BSRQ graphs: architecture, transfer coverage, loss/backward, and fuse paths."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace


STAGES = tuple(f"S{index}" for index in range(8))
EXPECTED_TYPES = {
    "S0": ("Conv", "FullPAD_Tunnel", "UDQDetect"),
    "S1": ("BoundedSpectralStem", "FullPAD_Tunnel", "UDQDetect"),
    "S2": ("Conv", "ReliabilityAdaptiveFullPAD", "UDQDetect"),
    "S3": ("Conv", "FullPAD_Tunnel", "SCQUDQDetect"),
    "S4": ("BoundedSpectralStem", "FullPAD_Tunnel", "SCQUDQDetect"),
    "S5": ("Conv", "ReliabilityAdaptiveFullPAD", "SCQUDQDetect"),
    "S6": ("BoundedSpectralStem", "ReliabilityAdaptiveFullPAD", "UDQDetect"),
    "S7": ("BoundedSpectralStem", "ReliabilityAdaptiveFullPAD", "SCQUDQDetect"),
}


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    return parser.parse_args()


def transferred_keys(network, source_state: dict) -> set[str]:
    target = network.state_dict()
    return {key for key, value in source_state.items() if key in target and value.shape == target[key].shape}


def main() -> None:
    args = parse_args()
    root, pretrained, report_dir = args.root.resolve(), args.pretrained.resolve(), args.report_dir.resolve()
    if not root.is_absolute() or not pretrained.is_absolute() or not report_dir.is_absolute():
        raise ValueError("root, pretrained, and report-dir must be absolute")
    if not pretrained.is_file():
        raise FileNotFoundError(pretrained)
    os.chdir("/tmp")  # Prove absolute-path workers still resolve the local project package.
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import torch
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.nn.tasks import torch_safe_load
    from ultralytics.utils import DEFAULT_CFG_DICT

    try:
        Path(ultralytics.__file__).resolve().relative_to(root)
    except ValueError as error:
        raise ImportError(f"Preflight imported external Ultralytics: {ultralytics.__file__}") from error

    checkpoint, _ = torch_safe_load(str(pretrained))
    source_model = checkpoint.get("ema") or checkpoint["model"]
    source_state = source_model.float().state_dict()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    coverage, fingerprints = {}, {}
    reference_coverage: set[str] | None = None
    for stage in STAGES:
        yaml_path = root / "ultralytics" / "cfg" / "models" / "v13" / f"yolov13n-bsrq-{stage.lower()}.yaml"
        if not yaml_path.is_file():
            raise FileNotFoundError(yaml_path)
        yolo = YOLO(str(yaml_path))
        network = yolo.model
        layers = network.model
        if len(layers) != 33:
            raise AssertionError(f"{stage}: expected 33 layers, got {len(layers)}")
        actual_types = (layers[0].__class__.__name__, layers[23].__class__.__name__, layers[-1].__class__.__name__)
        if actual_types != EXPECTED_TYPES[stage]:
            raise AssertionError(f"{stage}: expected {EXPECTED_TYPES[stage]}, got {actual_types}")
        head = layers[-1]
        if list(head.f) != [23, 27, 31] or list(map(int, head.stride.tolist())) != [8, 16, 32]:
            raise AssertionError(f"{stage}: invalid Detect inputs or strides: {head.f}, {head.stride}")
        if float(network.yaml.get("quality_gain", 0.0)) != 0.25:
            raise AssertionError(f"{stage}: quality_gain changed")
        keys = transferred_keys(network, source_state)
        if stage == "S0":
            reference_coverage = keys
        elif reference_coverage is None or not reference_coverage.issubset(keys):
            missing = sorted((reference_coverage or set()) - keys)
            raise AssertionError(f"{stage}: pretrained coverage fell below S0; missing {missing[:8]}")
        coverage[stage] = {
            "source_parameter_count": len(source_state),
            "target_parameter_count": len(network.state_dict()),
            "compatible_parameter_count": len(keys),
            "missing_from_s0_coverage": sorted((reference_coverage or set()) - keys),
            "has_inherited_stem_keys": all(key in keys for key in ("model.0.conv.weight", "model.0.bn.weight", "model.0.bn.bias")),
            "has_inherited_fullpad_gate": "model.23.gate" in keys,
        }
        if not coverage[stage]["has_inherited_stem_keys"] or not coverage[stage]["has_inherited_fullpad_gate"]:
            raise AssertionError(f"{stage}: B1 inherited stem/FullPAD coverage was lost")
        yolo.load(str(pretrained))
        network.to(device)
        image = torch.rand(1, 3, 640, 640, device=device)
        network.eval()
        with torch.no_grad():
            prediction = network(image)
        public_prediction = prediction[0] if isinstance(prediction, tuple) else prediction
        if public_prediction.ndim != 3 or public_prediction.shape[1] != 4 + int(head.nc):
            raise AssertionError(f"{stage}: invalid public inference shape {tuple(public_prediction.shape)}")
        network.train()
        batch = {
            "img": image,
            "batch_idx": torch.zeros(1, dtype=torch.long, device=device),
            "cls": torch.zeros(1, dtype=torch.float32, device=device),
            "bboxes": torch.tensor([[0.5, 0.5, 0.25, 0.25]], dtype=torch.float32, device=device),
        }
        # Trainer normally installs an attribute-style configuration; direct loss preflight must mirror it.
        network.args = SimpleNamespace(**DEFAULT_CFG_DICT)
        loss, loss_items = network.loss(batch)
        if not torch.isfinite(loss) or not torch.isfinite(loss_items).all():
            raise AssertionError(f"{stage}: non-finite training loss")
        loss.backward()
        network.eval().fuse(verbose=False)
        with torch.no_grad():
            fused_prediction = network(image)
        fused_public = fused_prediction[0] if isinstance(fused_prediction, tuple) else fused_prediction
        if fused_public.shape != public_prediction.shape or not torch.isfinite(fused_public).all():
            raise AssertionError(f"{stage}: fused inference failed")
        fingerprints[stage] = {
            "yaml": str(yaml_path),
            "yaml_sha256": sha256(yaml_path),
            "layer_count": len(layers),
            "anchor_types": actual_types,
            "detect_inputs": list(head.f),
            "strides": [int(value) for value in head.stride.tolist()],
            "quality_gain": float(network.yaml["quality_gain"]),
            "parameters": sum(parameter.numel() for parameter in network.parameters()),
            "public_inference_shape": list(public_prediction.shape),
            "fused_inference_shape": list(fused_public.shape),
        }
        del yolo, network, image, batch, prediction, fused_prediction
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    atomic_json(report_dir / "pretrained_coverage.json", coverage)
    atomic_json(
        report_dir / "architecture_fingerprints.json",
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "project_root": str(root),
            "pretrained": str(pretrained),
            "pretrained_sha256": sha256(pretrained),
            "ultralytics_path": str(Path(ultralytics.__file__).resolve()),
            "device": str(device),
            "stages": fingerprints,
        },
    )
    print("BSRQ full model preflight passed for S0--S7")


if __name__ == "__main__":
    main()
