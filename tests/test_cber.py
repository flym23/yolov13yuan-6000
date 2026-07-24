"""Unit and parser regression tests for consensus-budgeted evidence routing."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from ultralytics import YOLO
from ultralytics.nn.modules import CBERSCPGDSC3k2, ConsensusBudgetedEvidenceRouter


def test_cber_router_shape_identity_and_alpha_gradient():
    module = ConsensusBudgetedEvidenceRouter(64, 128, release_rho=0.20, local_self=0.75, co_weight=0.50).train()
    shallow = torch.randn(2, 64, 161, 159, requires_grad=True)
    semantic = torch.randn(2, 128, 81, 80, requires_grad=True)
    output, micro = module(shallow, semantic)
    assert output.shape == semantic.shape and micro.shape == (2, 1, 81, 80)
    assert torch.allclose(output, semantic, atol=1e-6, rtol=1e-6)
    assert torch.isfinite(output).all() and torch.isfinite(micro).all()
    output.square().mean().backward()
    assert module.alpha_raw.grad is not None and torch.isfinite(module.alpha_raw.grad).all()
    assert module.alpha_raw.grad.abs().sum() > 0


def test_cber_budget_mean_is_bounded_and_flat_background_has_zero_support():
    module = ConsensusBudgetedEvidenceRouter(32, 64, release_rho=0.20).eval()
    low, contrast, semantic_detail = (torch.randn(3, 64, 40, 40) for _ in range(3))
    with torch.no_grad():
        gate = module._budgeted_channel_gate(low, contrast, semantic_detail)
    assert gate.shape == (3, 64, 1, 1) and torch.isfinite(gate).all()
    assert gate.min() >= 0.0 and gate.max() <= 1.0
    assert (gate.float().mean(dim=1) <= 0.20 + 1e-5).all()
    detail = torch.zeros(2, 1, 20, 20)
    assert torch.allclose(module._relative_support(detail, torch.zeros_like(detail)), torch.zeros_like(detail), atol=1e-7, rtol=0.0)


def test_cber_compatibility_gradients_and_rho_zero_endpoint():
    module = ConsensusBudgetedEvidenceRouter(32, 64, release_rho=0.20).train()
    with torch.no_grad():
        module.alpha_raw.fill_(0.10)
    output, _ = module(torch.randn(2, 32, 80, 80), torch.randn(2, 64, 40, 40))
    output.square().mean().backward()
    assert module.spatial_compatibility[-1].weight.grad is not None
    assert module.channel_compatibility[-1].weight.grad is not None
    assert torch.isfinite(module.spatial_compatibility[-1].weight.grad).all()
    assert torch.isfinite(module.channel_compatibility[-1].weight.grad).all()
    geometry_only = ConsensusBudgetedEvidenceRouter(32, 64, release_rho=0.0).train()
    with torch.no_grad():
        geometry_only.alpha_raw.fill_(0.50)
    semantic = torch.randn(2, 64, 40, 40)
    output, _ = geometry_only(torch.randn(2, 32, 80, 80), semantic)
    assert torch.allclose(output, semantic, atol=1e-6, rtol=1e-6)


def test_cber_block_shape_and_loss_for_empty_and_normal_gt():
    module = CBERSCPGDSC3k2(64, 64, 128, n=1, dsc3k=False, release_rho=0.20, local_self=0.75, co_weight=0.50)
    assert module([torch.randn(2, 64, 160, 160), torch.randn(2, 64, 80, 80)]).shape == (2, 128, 80, 80)
    model = YOLO("ultralytics/cfg/models/v13/yolov13-cber-n4-full.yaml").model.train()
    model.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)
    for with_target in (False, True):
        batch = {
            "img": torch.randn(1, 3, 128, 128),
            "batch_idx": torch.tensor([0], dtype=torch.long) if with_target else torch.empty(0, dtype=torch.long),
            "cls": torch.tensor([[0.0]]) if with_target else torch.empty((0, 1)),
            "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]) if with_target else torch.empty((0, 4)),
        }
        loss, _ = model.loss(batch)
        assert torch.isfinite(loss)
        loss.backward()


@pytest.mark.parametrize("scale", ("n", "s", "l", "x"))
def test_cber_n_s_l_x_parser_and_detect_stride(scale):
    model = YOLO(f"ultralytics/cfg/models/v13/yolov13{scale}-cber-n4-full.yaml").model
    assert isinstance(model.model[4], CBERSCPGDSC3k2)
    assert list(model.model[32].f) == [23, 27, 31]
    assert torch.allclose(model.model[32].stride.cpu(), torch.tensor([8.0, 16.0, 32.0]))
