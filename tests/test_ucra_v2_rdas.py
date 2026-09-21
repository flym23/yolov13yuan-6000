import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import ultralytics
assert Path(ultralytics.__file__).resolve().is_relative_to(ROOT)

import torch
import torch.nn.functional as F

from ultralytics.nn.modules.ucra_v2 import UCRA1v2, UCRA2v2


def _assert_finite(tensor):
    assert torch.isfinite(tensor).all()


def test_exact_identity_start():
    torch.manual_seed(7)

    u1 = UCRA1v2(256, 128)
    x5 = torch.randn(2, 256, 20, 20)
    p4 = torch.randn(2, 128, 40, 40)
    y1 = u1([x5, p4])
    ref1 = F.interpolate(x5, size=(40, 40), mode="nearest")
    assert torch.equal(y1, ref1)

    u2 = UCRA2v2(128, 128)
    x4 = torch.randn(2, 128, 40, 40)
    p3 = torch.randn(2, 128, 80, 80)
    y2 = u2([x4, p3])
    ref2 = F.interpolate(x4, size=(80, 80), mode="nearest")
    assert torch.equal(y2, ref2)


def test_backward_and_shapes():
    torch.manual_seed(11)

    u1 = UCRA1v2(64, 32)
    deep = torch.randn(2, 64, 10, 10, requires_grad=True)
    lateral = torch.randn(2, 32, 20, 20, requires_grad=True)
    target = torch.randn(2, 64, 20, 20)

    output = u1([deep, lateral])
    loss = F.l1_loss(output, target)
    loss.backward()

    assert output.shape == target.shape
    _assert_finite(output)
    _assert_finite(deep.grad)
    assert u1.residual_out.weight.grad is not None
    assert u1.residual_out.weight.grad.abs().sum() > 0


def test_offset_bound():
    torch.manual_seed(13)

    module = UCRA2v2(
        64,
        32,
        max_offset=0.50,
    )

    with torch.no_grad():
        torch.nn.init.normal_(
            module.offset_head[-1].weight,
            std=0.5,
        )
        torch.nn.init.normal_(
            module.offset_head[-1].bias,
            std=0.5,
        )

    deep = torch.randn(2, 64, 10, 10)
    lateral = torch.randn(2, 32, 20, 20)
    parts = module.compute_components(deep, lateral)

    assert parts["offsets"].abs().max() <= 0.500001


def test_residual_budget_and_orthogonality():
    torch.manual_seed(17)

    semantic = UCRA1v2(
        64,
        32,
        max_residual_ratio=0.08,
    )
    detail = UCRA2v2(
        64,
        32,
        max_residual_ratio=0.10,
    )

    with torch.no_grad():
        torch.nn.init.normal_(
            semantic.residual_out.weight,
            std=0.10,
        )
        torch.nn.init.normal_(
            detail.residual_out.weight,
            std=0.10,
        )

    deep = torch.randn(2, 64, 10, 10)
    lateral = torch.randn(2, 32, 20, 20)

    sp = semantic.compute_components(deep, lateral)
    base = sp["base"].float()
    corr = sp["correction"].float()

    base_rms = base.square().mean(
        dim=(2, 3),
        keepdim=True,
    ).sqrt()
    corr_rms = corr.square().mean(
        dim=(2, 3),
        keepdim=True,
    ).sqrt()

    assert (
        corr_rms
        <= 0.08001 * base_rms + 1e-6
    ).all()

    dp = detail.compute_components(deep, lateral)
    base = dp["base"].float()
    corr = dp["correction"].float()

    base_rms = base.square().mean(
        dim=(1, 2, 3),
        keepdim=True,
    ).sqrt()
    corr_rms = corr.square().mean(
        dim=(1, 2, 3),
        keepdim=True,
    ).sqrt()

    assert (
        corr_rms
        <= 0.10001 * base_rms + 1e-6
    ).all()

    dot = (
        base * corr
    ).sum(dim=1, keepdim=True)

    assert dot.abs().max() < 1e-4


def test_strict_scale_error():
    module = UCRA1v2(64, 32)
    deep = torch.randn(1, 64, 10, 10)
    lateral = torch.randn(1, 32, 19, 20)

    failed = False
    try:
        module([deep, lateral])
    except ValueError:
        failed = True

    assert failed


if __name__ == "__main__":
    test_exact_identity_start()
    test_backward_and_shapes()
    test_offset_bound()
    test_residual_budget_and_orthogonality()
    test_strict_scale_error()
    print("All UCRA-v2 RDAS tests passed.")
