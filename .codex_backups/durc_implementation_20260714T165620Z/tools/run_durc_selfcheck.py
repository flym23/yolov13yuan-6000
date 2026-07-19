#!/usr/bin/env python3
"""Dependency-light DURC runtime self-check for training servers without pytest."""

import argparse
import sys
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ultralytics.nn.modules.block import DRSC, DRSCDSC3k2, DSC3k2  # noqa: E402
from ultralytics.nn.modules.head import Detect, HRCTDetect, NonUniformDFL  # noqa: E402
from ultralytics.utils.loss import DFLoss, SUDLBboxLoss, SUDLDFLoss  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA self-check requested but CUDA is unavailable")
    torch.manual_seed(0)

    drsc = DRSC(16).to(device)
    x = torch.randn(2, 16, 17, 19, device=device, requires_grad=True)
    output = drsc(x)
    assert output.shape == x.shape and torch.allclose(output, x, atol=1e-6, rtol=1e-6)
    output.square().mean().backward()
    assert drsc.alpha_raw.grad is not None and torch.isfinite(drsc.alpha_raw.grad).all()
    constant = torch.full((2, 16, 11, 13), 2.0, device=device)
    assert torch.allclose(constant, drsc._lowpass(constant), atol=1e-6, rtol=0.0)

    original_block = DSC3k2(16, 32, n=2, e=0.25)
    wrapped_block = DRSCDSC3k2(16, 32, n=2, e=0.25)
    incompatible = wrapped_block.load_state_dict(original_block.state_dict(), strict=False)
    assert not incompatible.unexpected_keys
    assert incompatible.missing_keys and all(key.startswith("drsc.") for key in incompatible.missing_keys)

    channels = (16, 32, 64)
    original_head = Detect(nc=4, ch=channels).to(device).train()
    hrct_head = HRCTDetect(nc=4, ch=channels).to(device).train()
    hrct_head.cv2.load_state_dict(original_head.cv2.state_dict())
    hrct_head.cv3.load_state_dict(original_head.cv3.state_dict())
    hrct_head.dfl.load_state_dict(original_head.dfl.state_dict())
    features = [
        torch.randn(2, 16, 20, 20, device=device),
        torch.randn(2, 32, 10, 10, device=device),
        torch.randn(2, 64, 5, 5, device=device),
    ]
    expected = original_head([feature.clone() for feature in features])
    actual = hrct_head([feature.clone() for feature in features])
    assert all(torch.allclose(a, b, atol=1e-6, rtol=1e-6) for a, b in zip(actual, expected))
    assert (hrct_head.hrct_p3.role, hrct_head.hrct_p4.role, hrct_head.hrct_p5.role) == (
        "detail",
        "balanced",
        "semantic",
    )

    projection = NonUniformDFL(16, gamma=1.5).to(device)
    logits = torch.randn(2, 64, 25, device=device, requires_grad=True)
    distances = projection(logits)
    assert distances.shape == (2, 4, 25) and torch.isfinite(distances).all()
    distances.mean().backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()

    uniform_project = torch.arange(16, dtype=torch.float32, device=device)
    sudl_dfl = SUDLDFLoss(uniform_project, use_soft_label=False).to(device)
    original_dfl = DFLoss(reg_max=16).to(device)
    dfl_logits = torch.randn(5, 4, 16, device=device)
    targets = torch.rand(5, 4, device=device) * 14.5
    scales = torch.full((5, 1), 32.0, device=device)
    assert torch.allclose(
        sudl_dfl(dfl_logits, targets, scales),
        original_dfl(dfl_logits.view(-1, 16), targets),
        atol=1e-6,
        rtol=1e-6,
    )

    bbox_loss = SUDLBboxLoss(reg_max=16, project=uniform_project).to(device)
    base = torch.tensor([[0.2], [0.5], [1.0]], device=device)
    uncertainty = torch.tensor([[0.0], [0.05], [0.5]], device=device)
    scale = torch.tensor([[8.0], [24.0], [64.0]], device=device)
    uncertainty_weight, _, extra = bbox_loss._extra_weights(uncertainty, scale, base)
    assert uncertainty_weight.min() >= 1.0
    assert uncertainty_weight.max() <= 1.0 + bbox_loss.uncertainty_gain
    assert torch.allclose((base * extra).sum(), base.sum(), atol=1e-6, rtol=1e-6)

    if device.type == "cuda":
        fp16 = DRSC(16).to(device).half()
        half_input = torch.randn(2, 16, 16, 16, device=device, dtype=torch.float16, requires_grad=True)
        half_output = fp16(half_input)
        assert torch.isfinite(half_output).all()
        half_output.float().mean().backward()
        assert half_input.grad is not None and torch.isfinite(half_input.grad).all()
    print(f"DURC runtime self-check passed on {device}")


if __name__ == "__main__":
    main()
