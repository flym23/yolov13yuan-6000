#!/usr/bin/env python3
"""Dependency-light two-round DCRQ module self-check for training hosts without pytest."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def ratio(base, residual):
    return float((residual.float().square().mean((2, 3)).sqrt() / base.float().square().mean((2, 3)).sqrt().clamp_min(1e-12)).max().detach())


def finite_gradients(module):
    gradients = [parameter.grad for parameter in module.parameters() if parameter.grad is not None]
    assert gradients and all(gradient.isfinite().all() for gradient in gradients)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    os.chdir("/tmp")
    sys.path.insert(0, str(root))
    import torch
    import ultralytics
    from ultralytics.nn.modules.dcrq import DAWRB, RCFConcat, RecallPreservingQualityCalibrator
    from ultralytics.utils.metrics import bbox_iou

    Path(ultralytics.__file__).resolve().relative_to(root)
    torch.manual_seed(20260914)

    for height, width in ((80, 80), (81, 79)):
        x = torch.randn(2, 16, height, width)
        output = DAWRB(16)(x)
        assert output.shape == x.shape and torch.equal(output, x) and torch.isfinite(output).all()
    deep, lateral = torch.randn(2, 16, 80, 80), torch.randn(2, 24, 80, 80)
    output = RCFConcat(16, 24)([deep, lateral])
    assert output.shape == (2, 40, 80, 80) and torch.equal(output, torch.cat((deep, lateral), dim=1))
    assert torch.equal(output[:, 16:], lateral)
    calibrator = RecallPreservingQualityCalibrator()
    cls = torch.full((1, 1, 1, 5), 0.4)
    quality = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0]).view(1, 1, 1, 5)
    calibrated = calibrator(cls, quality, 0).flatten()
    assert torch.all(calibrated[1:] >= calibrated[:-1])
    assert abs(float(calibrated[0] / 0.4) - 0.96875) < 1e-6 and abs(float(calibrated[-1] / 0.4) - 1.125) < 1e-6
    matched_iou = bbox_iou(
        torch.tensor([[0.0, 0.0, 2.0, 2.0], [1.0, 1.0, 3.0, 3.0]]),
        torch.tensor([[0.0, 0.0, 2.0, 2.0], [1.5, 1.5, 3.5, 3.5]]),
        xywh=False,
        CIoU=False,
    )
    quality_target = torch.zeros(2, 1)
    quality_target[:] = matched_iou.clamp(0.0, 1.0).reshape(-1, 1)
    assert matched_iou.shape == quality_target.shape == (2, 1) and torch.isfinite(quality_target).all()
    print("round1 identity/shape/finite/RPQC: PASS")

    daw = DAWRB(16)
    x = torch.randn(2, 16, 41, 39, requires_grad=True)
    torch.nn.init.normal_(daw.band_projection.weight, mean=0.0, std=0.02)
    output = daw(x)
    assert ratio(x, output - x) <= 0.05002
    output.float().mean().backward()
    assert torch.isfinite(x.grad).all()
    finite_gradients(daw)
    rcf = RCFConcat(16, 24)
    deep, lateral = torch.randn(2, 16, 40, 40, requires_grad=True), torch.randn(2, 24, 40, 40, requires_grad=True)
    torch.nn.init.normal_(rcf.out_projection.weight, mean=0.0, std=0.02)
    output = rcf([deep, lateral])
    assert ratio(deep, output[:, :16] - deep) <= 0.04002 and torch.equal(output[:, 16:], lateral)
    output.float().mean().backward()
    assert torch.isfinite(deep.grad).all() and torch.isfinite(lateral.grad).all()
    finite_gradients(rcf)
    print("round2 residual-budget/backward/finite: PASS")


if __name__ == "__main__":
    main()
