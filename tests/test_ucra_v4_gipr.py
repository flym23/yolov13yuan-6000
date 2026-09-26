import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from ultralytics.nn.modules.ucra_v2 import UCRA2v2
from ultralytics.nn.modules.ucra_v4 import UCRA2v4


def _finite(x):
    assert torch.isfinite(x).all()


def round1_structure_and_math():
    torch.manual_seed(123)
    rng = torch.random.get_rng_state().clone()
    base = UCRA2v2(64, 32)
    assert torch.equal(rng, torch.random.get_rng_state())
    v4 = UCRA2v4(64, 32)
    assert torch.equal(rng, torch.random.get_rng_state())

    # Exact inherited A4 parameters and exact A4 output at initialization.
    bp = dict(base.named_parameters())
    vp = dict(v4.named_parameters())
    for name, param in bp.items():
        assert name in vp
        assert torch.equal(param, vp[name]), name

    deep = torch.randn(2, 64, 10, 10)
    lateral = torch.randn(2, 32, 20, 20)
    assert torch.equal(base([deep, lateral]), v4([deep, lateral]))

    # New precision geometry must be a pure differential source.
    zero_offset = UCRA2v4(64, 32, max_precision_offset=0.0)
    parts = zero_offset.compute_components(
        torch.randn(1, 64, 10, 10),
        torch.randn(1, 32, 20, 20),
    )
    assert torch.equal(parts["precision_offsets"], torch.zeros_like(parts["precision_offsets"]))
    assert torch.equal(parts["precision_geometry"], torch.zeros_like(parts["precision_geometry"]))

    constant = UCRA2v4(64, 32)
    parts = constant.compute_components(
        torch.ones(1, 64, 10, 10),
        torch.randn(1, 32, 20, 20),
    )
    assert parts["precision_geometry"].abs().max() < 1e-6
    assert parts["precision_offsets"].abs().max() <= constant.max_precision_offset + 1e-6

    # Force the branch open and verify hard safety constraints.
    active = UCRA2v4(
        64,
        32,
        max_precision_gain=1.0,
        max_precision_ratio=0.04,
        precision_spatial_rho=0.15,
    )
    with torch.no_grad():
        active.precision_gain_raw.fill_(3.0)
        active.precision_offset_head[-1].weight.normal_(0.0, 0.5)
        active.precision_out.weight.normal_(0.0, 0.3)

    parts = active.compute_components(
        torch.randn(2, 64, 10, 10),
        torch.randn(2, 32, 20, 20),
    )
    ref = parts["core_output"].float()
    corr = parts["precision_correction"].float()
    ref_rms = ref.square().mean((1, 2, 3), keepdim=True).sqrt()
    corr_rms = corr.square().mean((1, 2, 3), keepdim=True).sqrt()
    ratio = corr_rms / (ref_rms + 1e-12)
    assert ratio.max() <= 0.04001

    dot = (ref * corr).sum(dim=1, keepdim=True)
    assert dot.abs().max() < 1e-4

    gate_mean = parts["precision_gate"].float().mean((2, 3))
    assert gate_mean.max() <= 0.15001

    print(
        "ROUND1_PASS",
        f"max_ratio={float(ratio.max().detach()):.8f}",
        f"max_abs_dot={float(dot.abs().max().detach()):.3e}",
        f"max_gate_mean={float(gate_mean.max().detach()):.6f}",
    )


def round2_training_and_gradient_isolation():
    torch.manual_seed(456)
    base = UCRA2v2(64, 32)
    v4 = UCRA2v4(64, 32)

    deep0 = torch.randn(2, 64, 10, 10)
    lateral0 = torch.randn(2, 32, 20, 20)
    target = torch.randn(2, 64, 20, 20)

    d0 = deep0.clone().requires_grad_(True)
    l0 = lateral0.clone().requires_grad_(True)
    d4 = deep0.clone().requires_grad_(True)
    l4 = lateral0.clone().requires_grad_(True)

    loss0 = F.mse_loss(base([d0, l0]), target)
    loss4 = F.mse_loss(v4([d4, l4]), target)
    assert torch.equal(loss0, loss4)
    loss0.backward()
    loss4.backward()

    # New branch must not disturb the inherited A4 gradient field at initialization.
    bp = dict(base.named_parameters())
    vp = dict(v4.named_parameters())
    for name, param in bp.items():
        g0 = param.grad
        g4 = vp[name].grad
        assert (g0 is None) == (g4 is None), name
        if g0 is not None:
            assert torch.equal(g0, g4), name

    assert torch.equal(d0.grad, d4.grad)
    assert torch.equal(l0.grad, l4.grad)

    # Zero-start gain gets the first signal, then adapter internals become trainable.
    assert v4.precision_gain_raw.grad is not None
    _finite(v4.precision_gain_raw.grad)
    first_gain_grad = float(v4.precision_gain_raw.grad.abs().detach())
    assert first_gain_grad > 0.0

    assert v4.precision_out.weight.grad is not None
    assert float(v4.precision_out.weight.grad.abs().sum().detach()) == 0.0
    assert v4.precision_offset_head[-1].weight.grad is not None
    assert float(v4.precision_offset_head[-1].weight.grad.abs().sum().detach()) == 0.0

    optimizer = torch.optim.SGD(v4.parameters(), lr=0.5)
    out_grad_seen = False
    offset_grad_seen = False
    for _ in range(4):
        optimizer.zero_grad(set_to_none=True)
        deep = torch.randn(2, 64, 10, 10)
        lateral = torch.randn(2, 32, 20, 20)
        target = torch.randn(2, 64, 20, 20)

        out = v4([deep, lateral])
        loss = F.mse_loss(out, target)
        loss.backward()
        _finite(out)
        _finite(loss)

        g_out = v4.precision_out.weight.grad
        g_off = v4.precision_offset_head[-1].weight.grad
        if g_out is not None and float(g_out.abs().sum().detach()) > 0.0:
            out_grad_seen = True
        if g_off is not None and float(g_off.abs().sum().detach()) > 0.0:
            offset_grad_seen = True
        optimizer.step()

    assert out_grad_seen
    assert offset_grad_seen

    failed = False
    try:
        v4([
            torch.randn(1, 64, 10, 10),
            torch.randn(1, 32, 19, 20),
        ])
    except ValueError:
        failed = True
    assert failed

    zero = v4([
        torch.zeros(1, 64, 10, 10),
        torch.zeros(1, 32, 20, 20),
    ])
    assert torch.equal(zero, torch.zeros_like(zero))

    print(
        "ROUND2_PASS",
        f"first_gain_grad={first_gain_grad:.3e}",
        f"out_grad_seen={out_grad_seen}",
        f"offset_grad_seen={offset_grad_seen}",
    )


if __name__ == "__main__":
    round1_structure_and_math()
    round2_training_and_gradient_isolation()
