"""Unit and integration checks for CMRF-YOLOv13 modules."""

import torch
import torch.nn.functional as F

from ultralytics.nn.modules.block import (
    CMRFDSC3k2,
    DSC3k2,
    ConsensusReliabilityRouter,
    ReliabilityFrequencyAlignUp,
    SymmetricMomentPreservedGeometry,
)


def _finite_grad(parameter):
    assert parameter.grad is not None
    assert torch.isfinite(parameter.grad).all()


def test_crr_identity_release_budget_and_second_stage_gradients():
    module = ConsensusReliabilityRouter(16, 32, release_rho=0.25).train().set_diagnostics(True)
    shallow = torch.randn(2, 16, 33, 31, requires_grad=True)
    semantic = torch.randn(2, 32, 17, 16, requires_grad=True)
    output, reliability = module(shallow, semantic)
    assert torch.allclose(output, semantic, atol=1e-6, rtol=1e-6)
    assert reliability.shape == (2, 1, 17, 16)
    assert reliability.isfinite().all() and 0.0 <= reliability.min() <= reliability.max() <= 1.0
    assert module.latest_diagnostics["channel_release"].float().mean(1).max() <= 0.25001
    output.square().mean().backward()
    _finite_grad(module.alpha_raw)
    with torch.no_grad():
        module.alpha_raw.fill_(0.2)
    module.zero_grad(set_to_none=True)
    routed, reliability = module(shallow, semantic)
    ((routed * torch.randn_like(semantic)).mean() + 0.01 * reliability.mean()).backward()
    for name, parameter in module.named_parameters():
        if name.startswith(("low_proj", "detail_proj", "route_gate", "spatial_gate", "channel_gate", "reliability_head", "out_proj")):
            _finite_grad(parameter)


def test_smpg_identity_moment_bounds_and_gradients():
    module = SymmetricMomentPreservedGeometry(32, max_radius=1.25).train().set_diagnostics(True)
    x = torch.randn(2, 32, 13, 11, requires_grad=True)
    output = module(x, torch.rand(2, 1, 13, 11))
    diag = module.latest_diagnostics
    assert torch.allclose(output, x, atol=1e-6, rtol=1e-6)
    assert diag["offsets"].shape == (2, 5, 2, 13, 11)
    assert diag["offsets"].abs().max() <= 1.250001
    assert torch.allclose(diag["weights"].sum(1), torch.ones_like(diag["weights"][:, 0]), atol=1e-6)
    assert diag["weights"][:, 0].min() >= 0.50 - 1e-6
    assert diag["weights"][:, 0].max() <= 0.80 + 1e-6
    assert diag["first_moment"].abs().max() <= 1e-6
    output.square().mean().backward()
    _finite_grad(module.alpha_raw)
    with torch.no_grad():
        module.alpha_raw.fill_(0.2)
    module.zero_grad(set_to_none=True)
    (module(x, torch.rand(2, 1, 13, 11)) * torch.randn_like(x)).mean().backward()
    for name in ("radius_head.weight", "pair_mass_head.weight", "center_head.weight", "angle_head.weight", "out_proj.weight"):
        _finite_grad(dict(module.named_parameters())[name])


def test_cmrf_identity_and_rng_pairing():
    module = CMRFDSC3k2(16, 16, 32, n=1, dsc3k=False, e=0.25).train()
    p2, p3 = torch.randn(2, 16, 32, 32), torch.randn(2, 16, 16, 16)
    assert torch.allclose(module([p2, p3]), super(CMRFDSC3k2, module).forward(p3), atol=1e-6, rtol=1e-6)
    torch.manual_seed(123)
    _ = DSC3k2(c1=16, c2=32, n=1, dsc3k=False, e=0.25)
    expected = torch.randn(16)
    torch.manual_seed(123)
    _ = CMRFDSC3k2(16, 16, 32, n=1, dsc3k=False, e=0.25)
    assert torch.equal(torch.randn(16), expected)


def test_rfa_identity_true_rms_budget_zero_base_and_rng_pairing():
    module = ReliabilityFrequencyAlignUp(32, 16, max_residual_ratio=0.15).train().set_diagnostics(True)
    deep = torch.randn(2, 32, 9, 8, requires_grad=True)
    lateral = torch.randn(2, 16, 18, 16, requires_grad=True)
    output = module([deep, lateral])
    base = F.interpolate(deep, size=lateral.shape[-2:], mode="nearest")
    assert torch.allclose(output, base, atol=1e-6, rtol=1e-6)
    _, correction, *_ = module.compute_components(deep, lateral)
    correction_rms = correction.float().square().mean((2, 3), keepdim=True).sqrt()
    base_rms = base.float().square().mean((2, 3), keepdim=True).sqrt()
    assert torch.all(correction_rms <= 0.15 * base_rms + 2e-6)
    _, zero_correction, *_ = module.compute_components(torch.zeros(2, 32, 3, 4), torch.randn(2, 16, 6, 8))
    assert zero_correction.abs().max().item() == 0.0
    output.square().mean().backward()
    _finite_grad(module.alpha_raw)
    with torch.no_grad():
        module.alpha_raw.fill_(0.2)
    module.zero_grad(set_to_none=True)
    (module([deep, lateral]) * torch.randn_like(base)).mean().backward()
    for parameter in module.parameters():
        _finite_grad(parameter)
    torch.manual_seed(456)
    expected = torch.randn(16)
    torch.manual_seed(456)
    _ = ReliabilityFrequencyAlignUp(16, 8)
    assert torch.equal(torch.randn(16), expected)


def test_full_yaml_structure_stride_and_fuse():
    from ultralytics.nn.tasks import DetectionModel

    model = DetectionModel("ultralytics/cfg/models/v13/yolov13-cmrf.yaml", ch=3, nc=4, verbose=False).eval()
    assert len(model.model) == 33
    assert model.model[4].__class__.__name__ == "CMRFDSC3k2"
    assert model.model[19].__class__.__name__ == "ReliabilityFrequencyAlignUp"
    assert model.model[20].__class__.__name__ == "Concat"
    assert list(model.stride.cpu().tolist()) == [8.0, 16.0, 32.0]
    with torch.no_grad():
        assert model(torch.randn(1, 3, 128, 128)) is not None
    model.fuse(verbose=False)
    with torch.no_grad():
        assert model(torch.randn(1, 3, 128, 128)) is not None
