"""Contract tests for GSD's independent SCGP, MCAS and CSTD innovations."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from ultralytics import YOLO
from ultralytics.nn.modules import CSTDDetect, MCASUp, SCPGDSC3k2, SemanticContextBridge


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "ultralytics" / "cfg" / "models" / "v13"


def test_mcas_initial_output_is_exact_nearest_and_preserves_rng():
    torch.manual_seed(123)
    before = torch.random.get_rng_state().clone()
    module = MCASUp(64, 32).eval()
    assert torch.equal(before, torch.random.get_rng_state())
    deep, lateral = torch.randn(2, 64, 20, 20), torch.randn(2, 32, 40, 40)
    with torch.no_grad():
        output = module([deep, lateral])
    assert torch.equal(output, F.interpolate(deep, size=lateral.shape[-2:], mode="nearest"))


def test_mcas_convex_weights_energy_budget_and_two_step_gradients():
    module = MCASUp(64, 32, max_residual_ratio=0.10).train()
    deep = torch.randn(2, 64, 20, 20, requires_grad=True)
    lateral = torch.randn(2, 32, 40, 40, requires_grad=True)
    with torch.no_grad():
        module.residual_out.weight.normal_(mean=0.0, std=1.0)
    base, correction, weights, _, _ = module.compute_components(deep, lateral)
    assert weights.shape == (2, 3, 40, 40)
    assert torch.allclose(weights.sum(dim=1), torch.ones_like(weights[:, 0]), atol=1e-6, rtol=1e-6)
    base_energy = base.float().square().mean(dim=(2, 3), keepdim=True).sqrt()
    correction_energy = correction.float().square().mean(dim=(2, 3), keepdim=True).sqrt()
    assert (correction_energy <= 0.10 * base_energy + 1e-5).all()
    # A fresh zero-start module has a two-step gradient path: output projection first, basis selector second.
    module = MCASUp(64, 32).train()
    deep = torch.randn(2, 64, 20, 20, requires_grad=True)
    lateral = torch.randn(2, 32, 40, 40, requires_grad=True)
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    module([deep, lateral]).square().mean().backward()
    assert module.residual_out.weight.grad is not None and module.residual_out.weight.grad.abs().sum() > 0
    assert module.deep_proj.weight.grad is None or module.deep_proj.weight.grad.abs().sum() == 0
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    module([deep, lateral]).square().mean().backward()
    assert module.deep_proj.weight.grad.abs().sum() > 0
    assert module.lateral_proj.weight.grad.abs().sum() > 0
    assert module.weight_predictor[0].weight.grad.abs().sum() > 0


def test_mcas_non_strict_odd_shape_and_context_zero_budget():
    module = MCASUp(64, 32, strict_scale=False).eval()
    output = module([torch.randn(1, 64, 17, 19), torch.randn(1, 32, 35, 39)])
    assert output.shape == (1, 64, 35, 39) and torch.isfinite(output).all()
    bridge = SemanticContextBridge(16, 32).eval()
    assert torch.equal(bridge._budget(torch.zeros(2, 16, 8, 8), torch.ones(2, 16, 8, 8)), torch.zeros(2, 16, 8, 8))


@pytest.mark.parametrize(
    "stage,flags",
    (
        ("t0-baseline", (False, False, False)), ("t1-scgp", (True, False, False)),
        ("t2-mcas", (False, True, False)), ("t3-cstd", (False, False, True)),
        ("t4-scgp-mcas", (True, True, False)), ("t5-scgp-cstd", (True, False, True)),
        ("t6-mcas-cstd", (False, True, True)), ("t7-full", (True, True, True)),
    ),
)
def test_gsd_full_factorial_parser(stage, flags):
    model = YOLO(str(MODEL_DIR / f"yolov13-gsd-{stage}.yaml")).model
    assert len(model.model) == 33
    assert isinstance(model.model[4], SCPGDSC3k2) is flags[0]
    assert isinstance(model.model[15], MCASUp) is flags[1]
    assert isinstance(model.model[32], CSTDDetect) is flags[2]
    assert list(model.model[32].f) == [23, 27, 31]
    assert torch.allclose(model.model[32].stride.cpu(), torch.tensor([8.0, 16.0, 32.0]))


@pytest.mark.parametrize("scale", ("n", "s", "l", "x"))
def test_gsd_scales_dynamic_shape_and_fuse(scale):
    model = YOLO(str(MODEL_DIR / f"yolov13{scale}-gsd.yaml")).model.eval()
    assert isinstance(model.model[4], SCPGDSC3k2)
    assert isinstance(model.model[15], MCASUp)
    assert isinstance(model.model[32], CSTDDetect)
    if scale == "n":
        with torch.no_grad():
            assert torch.isfinite(model(torch.randn(1, 3, 128, 128))[0]).all()
            assert torch.isfinite(model(torch.randn(1, 3, 160, 160))[0]).all()
        model.fuse(verbose=False)
        with torch.no_grad():
            assert torch.isfinite(model(torch.randn(1, 3, 128, 128))[0]).all()


def test_gsd_loss_for_empty_and_normal_gt():
    model = YOLO(str(MODEL_DIR / "yolov13-gsd-t7-full.yaml")).model.train()
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
