import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import ultralytics
assert Path(ultralytics.__file__).resolve().is_relative_to(ROOT)

import torch
import torch.nn.functional as F

from ultralytics.nn.modules.ucra_v3 import UCRA1v3, UCRA2v3


def _finite(x):
    assert torch.isfinite(x).all()


def test_round1_functional_invariants():
    torch.manual_seed(42)

    u1 = UCRA1v3(256, 128)
    deep1 = torch.randn(2, 256, 20, 20)
    lat1 = torch.randn(2, 128, 40, 40)
    out1 = u1([deep1, lat1])
    ref1 = F.interpolate(deep1, size=(40, 40), mode="nearest")
    assert torch.equal(out1, ref1)

    u2 = UCRA2v3(128, 128)
    deep2 = torch.randn(2, 128, 40, 40)
    lat2 = torch.randn(2, 128, 80, 80)
    out2 = u2([deep2, lat2])
    ref2 = F.interpolate(deep2, size=(80, 80), mode="nearest")
    assert torch.equal(out2, ref2)

    # Critical v2->v3 fix: zero displacement means zero geometric source.
    u20 = UCRA2v3(64, 32, max_offset=0.0)
    parts = u20.compute_components(
        torch.randn(1, 64, 10, 10),
        torch.randn(1, 32, 20, 20),
    )
    assert torch.equal(parts["offsets"], torch.zeros_like(parts["offsets"]))
    assert torch.equal(parts["geometry_source"], torch.zeros_like(parts["geometry_source"]))

    # A constant deep feature has no central-difference geometry even with offsets.
    uc = UCRA2v3(64, 32, max_offset=0.35)
    with torch.no_grad():
        uc.offset_head[-1].weight.normal_(0.0, 0.5)
        uc.offset_head[-1].bias.normal_(0.0, 0.5)
    parts = uc.compute_components(
        torch.ones(1, 64, 10, 10),
        torch.randn(1, 32, 20, 20),
    )
    assert parts["geometry_source"].abs().max() < 1e-6
    assert parts["offsets"].abs().max() <= 0.350001


def test_round1_budget_and_orthogonality():
    torch.manual_seed(17)

    u1 = UCRA1v3(64, 32, max_residual_ratio=0.06)
    with torch.no_grad():
        u1.semantic_out.weight.normal_(0.0, 0.2)
    parts1 = u1.compute_components(
        torch.randn(2, 64, 10, 10),
        torch.randn(2, 32, 20, 20),
    )
    base1 = parts1["base"].float()
    corr1 = parts1["correction"].float()
    base_rms1 = base1.square().mean((2, 3), keepdim=True).sqrt()
    corr_rms1 = corr1.square().mean((2, 3), keepdim=True).sqrt()
    assert (corr_rms1 <= 0.06001 * base_rms1 + 1e-6).all()

    u2 = UCRA2v3(64, 32, max_residual_ratio=0.08)
    with torch.no_grad():
        u2.detail_out.weight.normal_(0.0, 0.2)
        u2.geom_gain_raw.fill_(1.0)
        u2.offset_head[-1].weight.normal_(0.0, 0.2)
    parts2 = u2.compute_components(
        torch.randn(2, 64, 10, 10),
        torch.randn(2, 32, 20, 20),
    )
    base2 = parts2["base"].float()
    corr2 = parts2["correction"].float()
    base_rms2 = base2.square().mean((1, 2, 3), keepdim=True).sqrt()
    corr_rms2 = corr2.square().mean((1, 2, 3), keepdim=True).sqrt()
    assert (corr_rms2 <= 0.08001 * base_rms2 + 1e-6).all()

    dot = (base2 * corr2).sum(dim=1, keepdim=True)
    assert dot.abs().max() < 1e-4


def test_round2_rng_gradient_and_validation():
    torch.manual_seed(1234)
    before = torch.random.get_rng_state().clone()
    _ = UCRA1v3(64, 32)
    assert torch.equal(before, torch.random.get_rng_state())

    before = torch.random.get_rng_state().clone()
    _ = UCRA2v3(64, 32)
    assert torch.equal(before, torch.random.get_rng_state())

    torch.manual_seed(7)
    module = UCRA2v3(64, 32)
    optimizer = torch.optim.SGD(module.parameters(), lr=0.2)

    offset_grad_seen = False
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        deep = torch.randn(2, 64, 10, 10)
        lateral = torch.randn(2, 32, 20, 20)
        target = torch.randn(2, 64, 20, 20)
        out = module([deep, lateral])
        loss = F.mse_loss(out, target)
        loss.backward()
        _finite(out)
        _finite(module.detail_out.weight.grad)
        assert module.detail_out.weight.grad.abs().sum() > 0
        assert module.geom_gain_raw.grad is not None
        _finite(module.geom_gain_raw.grad)
        if step >= 1:
            grad = module.offset_head[-1].weight.grad
            if grad is not None and grad.abs().sum() > 0:
                offset_grad_seen = True
        optimizer.step()
    assert offset_grad_seen

    semantic = UCRA1v3(64, 32)
    out = semantic([
        torch.randn(2, 64, 10, 10),
        torch.randn(2, 32, 20, 20),
    ])
    out.abs().mean().backward()
    assert semantic.semantic_out.weight.grad is not None
    _finite(semantic.semantic_out.weight.grad)
    assert semantic.semantic_out.weight.grad.abs().sum() > 0

    failed = False
    try:
        module([
            torch.randn(1, 64, 10, 10),
            torch.randn(1, 32, 19, 20),
        ])
    except ValueError:
        failed = True
    assert failed

    z1 = UCRA1v3(64, 32)([
        torch.zeros(1, 64, 10, 10),
        torch.zeros(1, 32, 20, 20),
    ])
    z2 = UCRA2v3(64, 32)([
        torch.zeros(1, 64, 10, 10),
        torch.zeros(1, 32, 20, 20),
    ])
    assert torch.equal(z1, torch.zeros_like(z1))
    assert torch.equal(z2, torch.zeros_like(z2))
