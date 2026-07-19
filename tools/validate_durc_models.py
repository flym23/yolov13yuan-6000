#!/usr/bin/env python3
"""Build, forward, migrate, and profile all reviewed DURC configurations."""

import argparse
import gc
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from durc_experiments import MAIN_FILES, MODEL_FILES  # noqa: E402
from ultralytics import YOLO  # noqa: E402
from ultralytics.nn.modules.head import HRCTDetect, SUDLDetect  # noqa: E402
from ultralytics.nn.tasks import DetectionModel  # noqa: E402
from ultralytics.utils.torch_utils import (
    get_flops,
    get_num_gradients,
    get_num_params,
    intersect_dicts,
)  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--skip-complexity", action="store_true")
    return parser.parse_args()


def scaled_alias(config_path: Path, scale: str) -> Path:
    if config_path.name == "yolov13.yaml":
        return config_path.with_name(f"yolov13{scale}.yaml")
    if not config_path.name.startswith("yolov13-"):
        raise ValueError(f"unexpected DURC config name: {config_path.name}")
    return config_path.with_name(
        config_path.name.replace("yolov13-", f"yolov13{scale}-", 1)
    )


def assert_finite(value):
    if torch.is_tensor(value):
        if not torch.isfinite(value).all():
            raise RuntimeError("model output contains NaN or Inf")
    elif isinstance(value, dict):
        for item in value.values():
            assert_finite(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            assert_finite(item)


def assert_transfer(source_state, target_model, prefixes):
    target_state = target_model.state_dict()
    transferred = intersect_dicts(source_state, target_state)
    target_model.load_state_dict(transferred, strict=False)
    for prefix in prefixes:
        matched = [key for key in transferred if key.startswith(prefix)]
        if not matched:
            raise RuntimeError(
                f"no pretrained parameters transferred for prefix {prefix}"
            )
    return len(transferred)


def measure_complexity(model):
    return (
        len(list(model.modules())),
        get_num_params(model),
        get_num_gradients(model),
        get_flops(model, imgsz=640),
    )


def ensure_loss_args(model):
    if not hasattr(model, "args") or isinstance(model.args, dict):
        model_args = {"box": 7.5, "cls": 0.5, "dfl": 1.5}
        model_args.update(getattr(model, "args", {}))
        model.args = SimpleNamespace(**model_args)


def build_loss_smoke(model, device, imgsz):
    smoke_size = min(imgsz, 160)
    model = model.to(device).train()
    ensure_loss_args(model)
    batch = {
        "img": torch.randn(2, 3, smoke_size, smoke_size, device=device),
        "batch_idx": torch.tensor([0, 1], device=device),
        "cls": torch.tensor([[0.0], [1.0]], device=device),
        "bboxes": torch.tensor(
            [[0.50, 0.50, 0.20, 0.20], [0.35, 0.40, 0.12, 0.16]], device=device
        ),
    }
    loss, items = model(batch)
    assert_finite((loss, items))
    loss.backward()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    if not gradients or not all(
        torch.isfinite(gradient).all() for gradient in gradients
    ):
        raise RuntimeError(
            "synthetic training loss produced missing or non-finite gradients"
        )


def main():
    args = parse_args()
    root = args.root.resolve()
    cfg_dir = root / "ultralytics/cfg/models/v13"
    pretrained = root / "yolov13n.pt"
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA validation requested but CUDA is unavailable")
    for path in (cfg_dir / "yolov13.yaml", pretrained):
        if not path.is_file():
            raise FileNotFoundError(path)

    original = DetectionModel(
        str(scaled_alias(cfg_dir / "yolov13.yaml", "n")), ch=3, nc=4, verbose=False
    )
    if original.stride.tolist() != [8.0, 16.0, 32.0]:
        raise RuntimeError(
            f"original YOLOv13 stride regression: {original.stride.tolist()}"
        )
    if args.skip_complexity:
        baseline_complexity = {
            "layers": None,
            "params": None,
            "gradients": None,
            "gflops_640": None,
        }
    else:
        layers, params, gradients, flops = measure_complexity(original)
        baseline_complexity = {
            "layers": layers,
            "params": params,
            "gradients": gradients,
            "gflops_640": flops,
        }
    ensure_loss_args(original)
    original_criterion = original.init_criterion()
    if type(original_criterion.bbox_loss).__name__ != "BboxLoss":
        raise RuntimeError("original YOLOv13 no longer uses the prototype BboxLoss")
    build_loss_smoke(original, device, args.imgsz)
    source_state = YOLO(str(pretrained)).model.float().state_dict()
    del original

    expected_first_channels = {"n": 16, "s": 32, "l": 64, "x": 96}
    main_results = {}
    for stage, filename in MAIN_FILES.items():
        config_path = cfg_dir / filename
        for scale in "nslx":
            model = YOLO(str(scaled_alias(config_path, scale))).model
            first_channels = model.model[0].conv.out_channels
            if first_channels != expected_first_channels[scale]:
                raise RuntimeError(
                    f"{stage}/{scale} first channels={first_channels}, expected {expected_first_channels[scale]}"
                )
            if model.stride.tolist() != [8.0, 16.0, 32.0]:
                raise RuntimeError(
                    f"{stage}/{scale} invalid strides: {model.stride.tolist()}"
                )
            if scale == "n":
                model = model.to(device).eval()
                sample = torch.randn(
                    args.batch, 3, args.imgsz, args.imgsz, device=device
                )
                with torch.no_grad():
                    output = model(sample)
                assert_finite(output)
                if stage == "S3":
                    build_loss_smoke(model, device, args.imgsz)
            main_results[f"{stage}_{scale}"] = {
                "params": sum(parameter.numel() for parameter in model.parameters()),
                "head": type(model.model[-1]).__name__,
                "stride": model.stride.tolist(),
                "first_channels": first_channels,
            }
            del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    experiment_complexity = {}
    for stage, filename in MODEL_FILES.items():
        config_path = cfg_dir / filename
        model = YOLO(str(scaled_alias(config_path, "n"))).model
        head = model.model[-1]
        if stage.startswith("B") and not isinstance(head, HRCTDetect):
            raise RuntimeError(f"{stage} did not build HRCTDetect")
        if stage.startswith("C") and not isinstance(head, SUDLDetect):
            raise RuntimeError(f"{stage} did not build SUDLDetect")
        transfer_prefixes = [
            "model.2.cv1.",
            "model.2.cv2.",
            "model.4.cv1.",
            "model.4.cv2.",
        ]
        if stage.startswith("B"):
            transfer_prefixes.extend(
                ("model.32.cv2.", "model.32.cv3.", "model.32.dfl.")
            )
        elif stage.startswith("C"):
            transfer_prefixes.extend(("model.32.cv2.", "model.32.cv3."))
            if (
                "model.32.dfl.project" not in model.state_dict()
                or "model.32.dfl.conv.weight" in model.state_dict()
            ):
                raise RuntimeError(f"{stage} SUDL DFL state keys are invalid")
        transferred = assert_transfer(source_state, model, transfer_prefixes)
        if args.skip_complexity:
            layers = params = gradients = flops = None
        else:
            layers, params, gradients, flops = measure_complexity(model)
        experiment_complexity[stage] = {
            "config": filename,
            "head": type(head).__name__,
            "transferred_tensors": transferred,
            "layers": layers,
            "params": params,
            "gradients": gradients,
            "gflops_640": flops,
        }
        del model
        gc.collect()

    output_path = args.output or root / "runs/durc_ablation/complexity.json"
    if not output_path.is_absolute():
        output_path = root / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "imgsz": args.imgsz,
        "batch": args.batch,
        "baseline": baseline_complexity,
        "main_scale_checks": main_results,
        "experiments": experiment_complexity,
    }
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(output_path)
    print(output_path)


if __name__ == "__main__":
    main()
