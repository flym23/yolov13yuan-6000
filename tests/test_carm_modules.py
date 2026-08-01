"""Repository-level tests for CARM-YOLOv13.

Copy to tests/test_carm_modules.py after integrating the modules and YAML.
"""

from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.block import (
    CARMDSC3k2,
    MACRDSC3k2,
    MicroObjectMomentRefiner,
    MissingAwareCandidateReactivator,
    OrthogonalComplementaryAlignUp,
)
from ultralytics.nn.modules.block import CMRFDSC3k2, DSC3k2


def _assert_finite(tensor):
    assert torch.isfinite(tensor).all()


def test_micro_identity_moment_budget_and_two_stage_gradients():
    torch.manual_seed(1)
    module = MicroObjectMomentRefiner(32, 64).train()
    shallow = torch.randn(2, 32, 63, 61, requires_grad=True)
    semantic = torch.randn(2, 64, 32, 31, requires_grad=True)

    output = module(shallow, semantic)
    assert output.shape == semantic.shape
    assert torch.allclose(output, semantic, atol=1e-6, rtol=1e-6)

    residual, gate, offsets, weights, _ = module.compute_components(shallow, semantic)
    assert residual.shape == semantic.shape
    assert gate.mean(dim=(2, 3)).max() <= module.spatial_rho + 1e-6
    assert torch.allclose(weights.sum(dim=1), torch.ones_like(weights[:, 0]), atol=1e-6, rtol=1e-6)
    assert weights[:, 0].min() >= module.center_floor - 1e-6
    assert weights[:, 0].max() <= module.center_ceiling + 1e-6
    first_moment = (offsets.float() * weights.unsqueeze(2).float()).sum(dim=1)
    assert first_moment.abs().max() <= 1e-6
    assert offsets.abs().max() <= module.max_radius + 1e-6

    output.mean().backward()
    assert module.alpha_raw.grad is not None
    assert module.alpha_raw.grad.abs().sum() > 0
    _assert_finite(module.alpha_raw.grad)

    module.zero_grad(set_to_none=True)
    shallow.grad = semantic.grad = None
    with torch.no_grad():
        module.alpha_raw.fill_(0.5)
    module(shallow, semantic).square().mean().backward()
    branch_grads = [
        parameter.grad
        for name, parameter in module.named_parameters()
        if name != "alpha_raw" and parameter.requires_grad
    ]
    assert any(
        grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
        for grad in branch_grads
    )


def test_carm_exact_c4_initialization_and_rng_pairing():
    torch.manual_seed(7)
    _ = CMRFDSC3k2(32, 32, 64, n=1)
    sentinel_a = nn.Conv2d(5, 5, 1).weight.detach().clone()

    torch.manual_seed(7)
    module = CARMDSC3k2(32, 32, 64, n=1).eval()
    sentinel_b = nn.Conv2d(5, 5, 1).weight.detach().clone()
    assert torch.equal(sentinel_a, sentinel_b)

    p2 = torch.randn(2, 32, 64, 64)
    p3 = torch.randn(2, 32, 32, 32)
    output = module([p2, p3])
    c4_output = super(CARMDSC3k2, module).forward([p2, p3])
    assert output.shape == (2, 64, 32, 32)
    assert torch.allclose(output, c4_output, atol=1e-6, rtol=1e-6)


def test_ocaf_identity_orthogonality_budget_zero_base_and_gradients():
    torch.manual_seed(11)
    _ = nn.Upsample(scale_factor=2, mode="nearest")
    sentinel_a = nn.Conv2d(3, 3, 1).weight.detach().clone()

    torch.manual_seed(11)
    module = OrthogonalComplementaryAlignUp(64, 48).train()
    sentinel_b = nn.Conv2d(3, 3, 1).weight.detach().clone()
    assert torch.equal(sentinel_a, sentinel_b)

    deep = torch.randn(2, 64, 15, 13, requires_grad=True)
    lateral = torch.randn(2, 48, 30, 26, requires_grad=True)
    output = module([deep, lateral])
    nearest = F.interpolate(deep, size=lateral.shape[-2:], mode="nearest")
    assert torch.allclose(output, nearest, atol=1e-6, rtol=1e-6)

    base, correction, gate, *_ = module.compute_components(deep, lateral)
    dot = (base.float() * correction.float()).sum(dim=1)
    denominator = base.float().norm(dim=1) * correction.float().norm(dim=1) + 1e-8
    assert (dot.abs() / denominator).max() <= 2e-4

    base_rms = base.float().square().mean(dim=(1, 2, 3), keepdim=True).sqrt()
    correction_rms = correction.float().square().mean(dim=(1, 2, 3), keepdim=True).sqrt()
    assert (correction_rms <= module.max_residual_ratio * base_rms + 2e-6).all()
    assert gate.min() >= 0.0 and gate.max() <= 1.0

    zero_base, zero_correction, *_ = module.compute_components(
        torch.zeros(1, 64, 7, 6), torch.randn(1, 48, 14, 12)
    )
    assert torch.count_nonzero(zero_base) == 0
    assert torch.count_nonzero(zero_correction) == 0

    output.mean().backward()
    assert module.alpha_raw.grad is not None and module.alpha_raw.grad.abs().sum() > 0
    module.zero_grad(set_to_none=True)
    deep.grad = lateral.grad = None
    with torch.no_grad():
        module.alpha_raw.fill_(0.5)
    module([deep, lateral]).square().mean().backward()
    branch_grads = [
        parameter.grad
        for name, parameter in module.named_parameters()
        if name != "alpha_raw" and parameter.requires_grad
    ]
    assert any(
        grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
        for grad in branch_grads
    )


def test_reactivator_identity_budgets_and_two_stage_gradients():
    torch.manual_seed(13)
    module = MissingAwareCandidateReactivator(32, 64, 48).train()
    shallow = torch.randn(2, 32, 63, 61, requires_grad=True)
    base = torch.randn(2, 48, 32, 31, requires_grad=True)
    context = torch.randn(2, 64, 16, 16, requires_grad=True)

    output = module(shallow, base, context)
    assert torch.allclose(output, base, atol=1e-6, rtol=1e-6)
    residual, spatial, channel, *_ = module.compute_components(shallow, base, context)
    assert spatial.mean(dim=(2, 3)).max() <= module.spatial_rho + 1e-6
    assert channel.mean(dim=1).max() <= module.channel_rho + 1e-6
    base_rms = base.float().square().mean(dim=(2, 3), keepdim=True).sqrt()
    residual_rms = residual.float().square().mean(dim=(2, 3), keepdim=True).sqrt()
    assert (residual_rms <= module.max_residual_ratio * base_rms + 2e-6).all()

    output.mean().backward()
    assert module.alpha_raw.grad is not None and module.alpha_raw.grad.abs().sum() > 0
    module.zero_grad(set_to_none=True)
    shallow.grad = base.grad = context.grad = None
    with torch.no_grad():
        module.alpha_raw.fill_(0.5)
    module(shallow, base, context).square().mean().backward()
    branch_grads = [
        parameter.grad
        for name, parameter in module.named_parameters()
        if name != "alpha_raw" and parameter.requires_grad
    ]
    assert any(
        grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
        for grad in branch_grads
    )


def test_macr_exact_original_initialization_and_rng_pairing():
    torch.manual_seed(17)
    _ = DSC3k2(128, 48, n=1, dsc3k=True)
    sentinel_a = nn.Conv2d(4, 4, 1).weight.detach().clone()

    torch.manual_seed(17)
    module = MACRDSC3k2(32, 128, 64, 48, n=1, dsc3k=True).eval()
    sentinel_b = nn.Conv2d(4, 4, 1).weight.detach().clone()
    assert torch.equal(sentinel_a, sentinel_b)

    shallow = torch.randn(2, 32, 64, 64)
    fused = torch.randn(2, 128, 32, 32)
    context = torch.randn(2, 64, 16, 16)
    output = module([shallow, fused, context])
    base_output = super(MACRDSC3k2, module).forward(fused)
    assert output.shape == (2, 48, 32, 32)
    assert torch.allclose(output, base_output, atol=1e-6, rtol=1e-6)


def test_flat_maps_do_not_create_false_support():
    micro = MicroObjectMomentRefiner(16, 32).eval()
    _, gate, *_ = micro.compute_components(
        torch.zeros(1, 16, 16, 16), torch.zeros(1, 32, 8, 8)
    )
    assert torch.count_nonzero(gate) == 0

    up = OrthogonalComplementaryAlignUp(32, 24).eval()
    _, correction, gate, _, _, detail, _ = up.compute_components(
        torch.zeros(1, 32, 4, 5), torch.zeros(1, 24, 8, 10)
    )
    assert torch.count_nonzero(correction) == 0
    assert torch.count_nonzero(detail) == 0
    assert torch.count_nonzero(gate) == 0

    react = MissingAwareCandidateReactivator(16, 32, 24).eval()
    residual, spatial, *_ = react.compute_components(
        torch.zeros(1, 16, 16, 16),
        torch.zeros(1, 24, 8, 8),
        torch.zeros(1, 32, 4, 4),
    )
    assert torch.count_nonzero(residual) == 0
    assert torch.count_nonzero(spatial) == 0


def test_carm_model_yaml_build_and_stride():
    from ultralytics.nn.tasks import DetectionModel

    yaml_path = Path("ultralytics/cfg/models/v13/yolov13-carm.yaml")
    if not yaml_path.exists():
        pytest.skip(f"Missing {yaml_path}")
    model = DetectionModel(str(yaml_path), ch=3, nc=4, verbose=False)
    assert len(model.model) == 33
    assert model.model[4].__class__.__name__ == "CARMDSC3k2"
    assert model.model[19].__class__.__name__ == "OrthogonalComplementaryAlignUp"
    assert model.model[20].__class__.__name__ == "Concat"
    assert model.model[21].__class__.__name__ == "MACRDSC3k2"
    assert model.model[32].__class__.__name__ == "Detect"
    assert list(model.stride.cpu().tolist()) == [8.0, 16.0, 32.0]


def test_model_eval_and_fuse_smoke():
    from ultralytics.nn.tasks import DetectionModel

    yaml_path = Path("ultralytics/cfg/models/v13/yolov13-carm.yaml")
    if not yaml_path.exists():
        pytest.skip(f"Missing {yaml_path}")
    model = DetectionModel(str(yaml_path), ch=3, nc=4, verbose=False).eval()
    image = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        output_before = model(image)
    model.fuse(verbose=False).eval()
    with torch.no_grad():
        output_after = model(image)
    assert output_before is not None and output_after is not None


def test_all_carm_ablation_yamls_topology_and_stride():
    """Build A0-A7 and verify that only layers 4/19/21 change."""
    from ultralytics.nn.tasks import DetectionModel

    root = Path("ultralytics/cfg/models/v13/carm_ablation")
    expected = {
        "a0": ("CMRFDSC3k2", "Upsample", "DSC3k2"),
        "a1": ("CARMDSC3k2", "Upsample", "DSC3k2"),
        "a2": ("CMRFDSC3k2", "OrthogonalComplementaryAlignUp", "DSC3k2"),
        "a3": ("CMRFDSC3k2", "Upsample", "MACRDSC3k2"),
        "a4": ("CARMDSC3k2", "OrthogonalComplementaryAlignUp", "DSC3k2"),
        "a5": ("CARMDSC3k2", "Upsample", "MACRDSC3k2"),
        "a6": ("CMRFDSC3k2", "OrthogonalComplementaryAlignUp", "MACRDSC3k2"),
        "a7": ("CARMDSC3k2", "OrthogonalComplementaryAlignUp", "MACRDSC3k2"),
    }

    for key, classes in expected.items():
        matches = sorted(root.glob(f"yolov13-carm-{key}-*.yaml"))
        assert len(matches) == 1, (key, matches)
        model = DetectionModel(str(matches[0]), ch=3, nc=4, verbose=False)
        assert len(model.model) == 33
        assert model.model[4].__class__.__name__ == classes[0]
        assert model.model[19].__class__.__name__ == classes[1]
        assert model.model[20].__class__.__name__ == "Concat"
        assert model.model[21].__class__.__name__ == classes[2]
        assert model.model[32].__class__.__name__ == "Detect"
        assert list(model.stride.cpu().tolist()) == [8.0, 16.0, 32.0]


def test_zero_base_budget_is_exact_for_macr():
    module = MissingAwareCandidateReactivator(16, 32, 24).eval()
    shallow = torch.randn(1, 16, 16, 16)
    base = torch.zeros(1, 24, 8, 8)
    context = torch.randn(1, 32, 4, 4)
    residual, *_ = module.compute_components(shallow, base, context)
    assert torch.count_nonzero(residual) == 0


def test_ocaf_sample_scalar_budget_preserves_orthogonality():
    module = OrthogonalComplementaryAlignUp(32, 24).eval()
    deep = torch.randn(2, 32, 7, 5)
    lateral = torch.randn(2, 24, 14, 10)
    base, correction, *_ = module.compute_components(deep, lateral)

    # é¢ç®ç¼©æ¾å¿é¡»æ¯æ¯ä¸ªæ ·æ¬ä¸ä¸ªæ éï¼ä¸è½æ¯éééç¼©æ¾ã
    _, raw_scale = module._energy_budget(base, correction)
    assert raw_scale.shape == (2, 1, 1, 1)

    dot = (base.float() * correction.float()).sum(dim=1)
    norm = base.float().norm(dim=1) * correction.float().norm(dim=1) + 1e-8
    assert (dot.abs() / norm).max() <= 2e-4
