"""Executable unit checks for BSRQ's identity, bound, inheritance, and gradient contracts."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from ultralytics.nn.modules.block import FullPAD_Tunnel
from ultralytics.nn.modules.bsrq import BoundedSpectralStem, ReliabilityAdaptiveFullPAD, SCQQualityFusion
from ultralytics.nn.modules.conv import Conv


def test_bounded_spectral_stem_inherits_b1_and_keeps_rng() -> None:
    torch.manual_seed(20260915)
    baseline = Conv(3, 16, 3, 2)
    torch.manual_seed(20260915)
    stem = BoundedSpectralStem(3, 16, 3, 2, 0.08, 8)
    baseline_state, stem_state = baseline.state_dict(), stem.state_dict()
    for key, value in baseline_state.items():
        assert torch.equal(value, stem_state[key]), f"BSS did not inherit exact B1 key {key}"
    baseline.eval()
    stem.eval()
    image = torch.rand(2, 3, 32, 32)
    assert torch.equal(baseline(image), stem(image)), "BSS must be an exact identity at initialization"
    with torch.no_grad():
        stem.spectral.mlp[-1].bias[0] = 0.5
    expected = stem.act(stem.conv(stem.spectral(image)))
    assert torch.allclose(stem.forward_fuse(image), expected), "BSS forward_fuse dropped the spectral branch"


def test_reliability_adaptive_fullpad_identity_and_bounds() -> None:
    torch.manual_seed(7)
    baseline, module = FullPAD_Tunnel(), ReliabilityAdaptiveFullPAD(0.20, 8, True)
    original = torch.randn(2, 32, 8, 8, requires_grad=True)
    enhanced = torch.randn(2, 32, 8, 8, requires_grad=True)
    assert torch.equal(baseline([original, enhanced]), module([original, enhanced]))
    with torch.no_grad():
        module.gate.fill_(0.7)
        module.modulator[-1].bias.fill_(5.0)
    output = module([original, enhanced])
    factor = (output - original) / (module.gate * enhanced)
    assert float(factor.min()) >= 0.8 - 1e-5 and float(factor.max()) <= 1.2 + 1e-5
    output.square().mean().backward()
    assert module.gate.grad is not None and torch.isfinite(module.gate.grad).all()
    assert module.modulator[-1].weight.grad is not None and torch.isfinite(module.modulator[-1].weight.grad).all()


def test_scq_quality_fusion_identity_bounds_and_backward() -> None:
    torch.manual_seed(8)
    fusion = SCQQualityFusion((1.0, 0.5, 0.25), (0.25, 0.15, 0.08), 8)
    quality_feature = torch.randn(2, 1, 10, 10, requires_grad=True)
    quality_statistic = torch.randn(2, 1, 10, 10, requires_grad=True)
    statistics = torch.randn(2, 12, 10, 10, requires_grad=True)
    actual = fusion(quality_feature, quality_statistic, statistics, 1)
    expected = quality_feature + 0.5 * quality_statistic
    assert torch.equal(actual, expected), "SCQ must reproduce B1 fusion at initialization"
    with torch.no_grad():
        fusion.reliability.gates[1][-1].bias.fill_(5.0)
    reliability = fusion.reliability(statistics, 1)
    assert float(reliability.min()) >= 0.85 - 1e-5 and float(reliability.max()) <= 1.15 + 1e-5
    fusion(quality_feature, quality_statistic, statistics, 1).square().mean().backward()
    assert quality_feature.grad is not None and quality_statistic.grad is not None
    assert statistics.grad is None, "SCQ reliability evidence must be detached"
    assert fusion.reliability.gates[1][-1].weight.grad is not None


def main() -> None:
    test_bounded_spectral_stem_inherits_b1_and_keeps_rng()
    test_reliability_adaptive_fullpad_identity_and_bounds()
    test_scq_quality_fusion_identity_bounds_and_backward()
    print("BSRQ module unit checks passed")


if __name__ == "__main__":
    main()
