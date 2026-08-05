"""Repository-level invariants for MESA-YOLOv13."""

from pathlib import Path

import torch

from ultralytics.nn.modules.block import (
    DualMomentBoundaryRefiner,
    GradientIsolatedScaleReactivator,
    HomeostaticEvidenceMassAllocator,
    MACRDSC3k2,
    MESADSC3k2,
)


def _kwargs():
    return dict(c_shallow=32, c_fused=128, c_context=64, c2=64, n=1, dsc3k=True, e=0.5)


def _inputs(requires_grad=False):
    return [
        torch.randn(2, 32, 63, 61, requires_grad=requires_grad),
        torch.randn(2, 128, 32, 31, requires_grad=requires_grad),
        torch.randn(2, 64, 16, 16, requires_grad=requires_grad),
    ]


def test_hema_mass_budget_zero_evidence_and_gradients():
    module = HomeostaticEvidenceMassAllocator(rho_min=0.04, rho_max=0.16).train()
    shallow, fused, context = _inputs()
    total, micro, medium, support = module(shallow, fused[:, :64], context)
    assert torch.allclose(total, micro + medium, atol=2e-6, rtol=0)
    assert total.float().mean((2, 3)).max() <= module.rho_max + 1e-6
    assert support.min() >= 0 and support.max() <= 1
    zeros = module(torch.zeros_like(shallow), torch.zeros_like(fused[:, :64]), torch.zeros_like(context))
    assert all(torch.count_nonzero(item) == 0 for item in zeros[:3])
    total.square().mean().backward()
    assert module.score_head[-1].weight.grad is not None


def test_gisr_isolates_context_gradient_and_keeps_zero_base_zero():
    module = GradientIsolatedScaleReactivator(32, 64, 64, large_protect_floor=0.0).train()
    shallow, _, context = _inputs(requires_grad=True)
    base = torch.randn(2, 64, 32, 31, requires_grad=True)
    gates = torch.rand(2, 1, 32, 31) * 0.1
    with torch.no_grad():
        module.alpha_raw.fill_(0.5)
    output = module(shallow, base, context, gates, gates, torch.ones_like(gates))
    output.square().mean().backward()
    assert context.grad is None or torch.count_nonzero(context.grad) == 0
    assert shallow.grad is not None and base.grad is not None
    zero = module(shallow.detach(), torch.zeros_like(base), context.detach(), gates, gates, torch.ones_like(gates))
    assert torch.count_nonzero(zero) == 0


def test_dmbr_is_identity_at_start_and_on_constant_features():
    module = DualMomentBoundaryRefiner(64).eval()
    feature, gate = torch.randn(2, 64, 31, 29), torch.rand(2, 1, 31, 29)
    assert torch.equal(module(feature, gate), feature)
    with torch.no_grad():
        module.alpha_raw.fill_(1.0)
    constant = torch.full((1, 64, 13, 11), 2.5)
    assert torch.allclose(module(constant, torch.ones(1, 1, 13, 11)), constant, atol=4e-6, rtol=0)


def test_mesa_is_exact_a3_at_initialization_and_checkpoint_compatible():
    torch.manual_seed(17)
    legacy = MACRDSC3k2(**_kwargs()).eval()
    torch.manual_seed(17)
    mesa = MESADSC3k2(**_kwargs()).eval()
    inputs = _inputs()
    assert torch.equal(legacy(inputs), mesa(inputs))
    loaded = MESADSC3k2(**_kwargs()).eval().load_state_dict(legacy.state_dict(), strict=False)
    assert loaded.unexpected_keys == []
    assert all(key.startswith(("mass_allocator.", "scale_reactivator.", "boundary_refiner.")) for key in loaded.missing_keys)


def test_mesa_all_switches_have_finite_two_stage_gradients():
    for flags in ((False, False, False), (True, False, False), (False, True, False), (False, False, True), (True, True, True)):
        model = MESADSC3k2(**_kwargs(), use_hema=flags[0], use_gisr=flags[1], use_dmbr=flags[2]).train()
        inputs = _inputs(requires_grad=True)
        output = model(inputs)
        assert output.shape == (2, 64, 32, 31) and torch.isfinite(output).all()
        output.mean().backward()
        model.zero_grad(set_to_none=True)
        for item in inputs:
            item.grad = None
        with torch.no_grad():
            model.scale_reactivator.alpha_raw.fill_(0.5)
            model.boundary_refiner.alpha_raw.fill_(0.5)
        model(inputs).square().mean().backward()
        assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_mesa_yamls_topology_stride_and_all_scales():
    from ultralytics.nn.tasks import DetectionModel

    root = Path("ultralytics/cfg/models/v13/mesa_ablation")
    files = sorted(root.glob("yolov13-mesa-m*.yaml"))
    assert len(files) == 8
    for path in files:
        model = DetectionModel(str(path), ch=3, nc=4, verbose=False)
        assert len(model.model) == 33
        assert model.model[4].__class__.__name__ == "CMRFDSC3k2"
        assert model.model[19].__class__.__name__ == "Upsample"
        assert model.model[20].__class__.__name__ == "Concat"
        assert model.model[21].__class__.__name__ in {"MACRDSC3k2", "MESADSC3k2"}
        assert list(model.stride.cpu().tolist()) == [8.0, 16.0, 32.0]
    for scale in "nslx":
        model = DetectionModel({**__import__("yaml").safe_load((root / "yolov13-mesa-m7-full.yaml").read_text(encoding="utf-8")), "scale": scale}, ch=3, nc=4, verbose=False)
        assert model.model[21].__class__.__name__ == "MESADSC3k2"
