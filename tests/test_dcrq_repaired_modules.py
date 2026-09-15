"""CPU tests that protect DCRQ repaired B3/B4 from layer-index and RNG regression."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn

from ultralytics.nn.modules.block import DSC3k2
from ultralytics.nn.modules.dcrq import DAWRBDSC3k2, RCFConcat


def test_dawrb_same_index_inheritance_identity_and_rng() -> None:
    torch.manual_seed(20260915)
    baseline = DSC3k2(64, 128, n=2, dsc3k=False, e=0.25)
    baseline_next = nn.Conv2d(128, 128, 1)
    torch.manual_seed(20260915)
    repaired = DAWRBDSC3k2(64, 128, n=2, dsc3k=False, e=0.25, dawrb_reduction=4, dawrb_max_residual_ratio=0.05)
    repaired_next = nn.Conv2d(128, 128, 1)
    inherited, repaired_state = baseline.state_dict(), repaired.state_dict()
    for key, value in inherited.items():
        assert key in repaired_state and torch.equal(value, repaired_state[key]), f"missing inherited B2 key: {key}"
    for key, value in baseline_next.state_dict().items():
        assert torch.equal(value, repaired_next.state_dict()[key]), f"DAWRB perturbed downstream RNG key: {key}"
    baseline.eval()
    repaired.eval()
    image = torch.randn(2, 64, 20, 20)
    assert torch.equal(baseline(image), repaired(image)), "zero-start DAWRB wrapper must equal B2 DSC3k2 at init"
    repaired(image).square().mean().backward()
    assert repaired.dawrb.band_projection.weight.grad is not None


def test_rcf_identity_and_rng_isolation() -> None:
    torch.manual_seed(20260916)
    reference_next = nn.Conv2d(32, 32, 1)
    torch.manual_seed(20260916)
    rcf = RCFConcat(16, 16, reduction=4, max_residual_ratio=0.04, detach_gate=True)
    candidate_next = nn.Conv2d(32, 32, 1)
    for key, value in reference_next.state_dict().items():
        assert torch.equal(value, candidate_next.state_dict()[key]), f"RCF perturbed downstream RNG key: {key}"
    deep = torch.randn(2, 16, 10, 10, requires_grad=True)
    lateral = torch.randn(2, 16, 10, 10, requires_grad=True)
    expected = torch.cat((deep, lateral), dim=1)
    actual = rcf([deep, lateral])
    assert torch.equal(actual, expected), "zero-start RCF must exactly reproduce Concat at init"
    actual.square().mean().backward()
    assert rcf.out_projection.weight.grad is not None


def main() -> None:
    test_dawrb_same_index_inheritance_identity_and_rng()
    test_rcf_identity_and_rng_isolation()
    print("DCRQ repaired module checks passed")


if __name__ == "__main__":
    main()
