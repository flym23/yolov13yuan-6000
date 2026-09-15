#!/usr/bin/env python3
"""Validate repaired DCRQ B3/B4 coverage and 640px model behavior before old-result cleanup."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace


MODELS = {
    "B2_D_RPQC": "yolov13n-dcrq-b2.yaml",
    "B3_DAWRB_D_RPQC": "yolov13n-dcrq-b3.yaml",
    "B4_DCRQ_FULL": "yolov13n-dcrq-b4.yaml",
}
EXPECTED = {
    "B2_D_RPQC": ("DSC3k2", "Concat"),
    "B3_DAWRB_D_RPQC": ("DAWRBDSC3k2", "Concat"),
    "B4_DCRQ_FULL": ("DAWRBDSC3k2", "RCFConcat"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root, pretrained, report_dir = args.root.resolve(), args.pretrained.resolve(), args.report_dir.resolve()
    if not root.is_absolute() or not pretrained.is_absolute() or not report_dir.is_absolute():
        raise ValueError("root, pretrained, and report-dir must be absolute")
    if not pretrained.is_file():
        raise FileNotFoundError(pretrained)
    os.chdir("/tmp")
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
    source = source_model.float().state_dict()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    coverage, fingerprints = {}, {}
    b2_keys: set[str] | None = None
    for stage, yaml_name in MODELS.items():
        yaml_path = root / "ultralytics" / "cfg" / "models" / "v13" / yaml_name
        yolo = YOLO(str(yaml_path))
        network, graph = yolo.model, yolo.model.model
        if len(graph) != 33 or (graph[4].__class__.__name__, graph[20].__class__.__name__) != EXPECTED[stage]:
            raise AssertionError(f"{stage}: repaired graph anchors are invalid")
        head = graph[-1]
        if list(head.f) != [23, 27, 31] or list(map(int, head.stride.tolist())) != [8, 16, 32]:
            raise AssertionError(f"{stage}: Detect inputs or stride changed")
        target = network.state_dict()
        compatible = {key for key, value in source.items() if key in target and value.shape == target[key].shape}
        if stage == "B2_D_RPQC":
            b2_keys = compatible
        elif compatible != b2_keys:
            missing, unexpected = sorted((b2_keys or set()) - compatible), sorted(compatible - (b2_keys or set()))
            raise AssertionError(f"{stage}: coverage must equal B2; missing={missing[:8]}, unexpected={unexpected[:8]}")
        if not all(key in compatible for key in ("model.4.cv1.conv.weight", "model.4.cv2.conv.weight")):
            raise AssertionError(f"{stage}: inherited model.4 DSC3k2 keys did not transfer")
        coverage[stage] = {
            "compatible_key_count": len(compatible),
            "compatible_key_sha256": hashlib.sha256("\n".join(sorted(compatible)).encode()).hexdigest(),
            "equal_to_b2": compatible == b2_keys,
        }
        yolo.load(str(pretrained))
        network.to(device)
        image = torch.rand(1, 3, 640, 640, device=device)
        network.eval()
        with torch.no_grad():
            prediction = network(image)
        public = prediction[0] if isinstance(prediction, tuple) else prediction
        if public.ndim != 3 or public.shape[1] != 4 + int(head.nc):
            raise AssertionError(f"{stage}: invalid public prediction shape {tuple(public.shape)}")
        network.train()
        network.args = SimpleNamespace(**DEFAULT_CFG_DICT)
        batch = {
            "img": image,
            "batch_idx": torch.zeros(1, dtype=torch.long, device=device),
            "cls": torch.zeros(1, dtype=torch.float32, device=device),
            "bboxes": torch.tensor([[0.5, 0.5, 0.25, 0.25]], dtype=torch.float32, device=device),
        }
        loss, loss_items = network.loss(batch)
        if not torch.isfinite(loss) or not torch.isfinite(loss_items).all():
            raise AssertionError(f"{stage}: non-finite loss")
        loss.backward()
        network.eval().fuse(verbose=False)
        with torch.no_grad():
            fused = network(image)
        fused_public = fused[0] if isinstance(fused, tuple) else fused
        if fused_public.shape != public.shape or not torch.isfinite(fused_public).all():
            raise AssertionError(f"{stage}: fused prediction is invalid")
        fingerprints[stage] = {
            "yaml": str(yaml_path),
            "yaml_sha256": sha256(yaml_path),
            "layer_count": len(graph),
            "anchor_types": [graph[4].__class__.__name__, graph[20].__class__.__name__, head.__class__.__name__],
            "detect_inputs": list(head.f),
            "strides": [int(value) for value in head.stride.tolist()],
            "parameters": sum(parameter.numel() for parameter in network.parameters()),
            "inference_shape": list(public.shape),
            "fused_shape": list(fused_public.shape),
        }
        del yolo, network, image, batch, prediction, fused
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    atomic_json(report_dir / "pretrained_coverage.json", coverage)
    atomic_json(
        report_dir / "architecture_fingerprints.json",
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "pretrained": str(pretrained),
            "pretrained_sha256": sha256(pretrained),
            "ultralytics_path": str(Path(ultralytics.__file__).resolve()),
            "device": str(device),
            "stages": fingerprints,
        },
    )
    print("DCRQ repaired B2/B3/B4 coverage and model preflight passed")


if __name__ == "__main__":
    main()
