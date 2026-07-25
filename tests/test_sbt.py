"""Contract tests for SBT's independent SCGP, BCRA, and CSTD innovations."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from ultralytics import YOLO
from ultralytics.nn.modules import BCRAUp, CSTDDetect, SCPGDSC3k2

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "ultralytics" / "cfg" / "models" / "v13"


def test_scgp_initially_matches_base_dsc3k2():
    module = SCPGDSC3k2(64, 64, 128, n=1, dsc3k=False, e=0.25).train()
    p2, p3 = torch.randn(2, 64, 160, 160), torch.randn(2, 64, 80, 80)
    output = module([p2, p3])
    base = super(SCPGDSC3k2, module).forward(p3)
    assert output.shape == (2, 128, 80, 80)
    assert torch.allclose(output, base, atol=1e-6, rtol=1e-6)


def test_bcra_exact_nearest_initialization_and_budget_contract():
    module = BCRAUp(64, 32).train()
    deep, lateral = torch.randn(2, 64, 20, 20), torch.randn(2, 32, 40, 40)
    output = module([deep, lateral])
    assert torch.equal(output, F.interpolate(deep, size=lateral.shape[-2:], mode="nearest"))
    base, correction, weights, confidence, _ = module.compute_components(deep, lateral)
    assert weights.shape == (2, 9, 40, 40) and confidence.shape == (2, 1, 40, 40)
    assert torch.allclose(weights.sum(dim=1), torch.ones_like(weights[:, 0]), atol=1e-5, rtol=1e-5)
    assert 0.0 <= confidence.min() and confidence.max() <= 1.0
    base_energy = base.float().square().mean(dim=(2, 3), keepdim=True).sqrt()
    correction_energy = correction.float().square().mean(dim=(2, 3), keepdim=True).sqrt()
    assert (correction_energy <= 0.20 * base_energy + 1e-5).all()
    output.square().mean().backward()
    assert module.residual_out.weight.grad is not None and module.residual_out.weight.grad.abs().sum() > 0


def test_cstd_initial_towers_and_two_step_context_gradients():
    head = CSTDDetect(nc=4, ch=(32, 64, 128)).train()
    features = [torch.randn(2, 32, 32, 32), torch.randn(2, 64, 16, 16), torch.randn(2, 128, 8, 8)]
    outputs = head(features)
    expected = [torch.cat((head.cv2[index](feature), head.cv3[index](feature)), dim=1) for index, feature in enumerate(features)]
    assert [tuple(output.shape) for output in outputs] == [(2, 68, 32, 32), (2, 68, 16, 16), (2, 68, 8, 8)]
    assert all(torch.equal(actual, reference) for actual, reference in zip(outputs, expected))
    optimizer = torch.optim.SGD(head.parameters(), lr=0.1)
    sum(output.square().mean() for output in outputs).backward()
    assert head.cls_p3_from_p4.out_proj.weight.grad is not None
    assert head.cls_p3_from_p4.out_proj.weight.grad.abs().sum() > 0
    optimizer.step()
    optimizer.zero_grad()
    sum(output.square().mean() for output in head(features)).backward()
    assert head.cls_p3_from_p4.context_proj.conv.weight.grad.abs().sum() > 0
    assert head.reg_p4_from_p3.source_proj.conv.weight.grad.abs().sum() > 0
    assert head.reg_p4_from_p3.edge_proj[0].weight.grad.abs().sum() > 0


@pytest.mark.parametrize(
    "stage,flags",
    (
        ("t0-baseline", (False, False, False)),
        ("t1-scgp", (True, False, False)),
        ("t2-bcra", (False, True, False)),
        ("t3-cstd", (False, False, True)),
        ("t4-scgp-bcra", (True, True, False)),
        ("t5-scgp-cstd", (True, False, True)),
        ("t6-bcra-cstd", (False, True, True)),
        ("t7-full", (True, True, True)),
    ),
)
def test_sbt_full_factorial_parser(stage, flags):
    model = YOLO(str(MODEL_DIR / f"yolov13-sbt-{stage}.yaml")).model
    assert len(model.model) == 33
    assert isinstance(model.model[4], SCPGDSC3k2) is flags[0]
    assert isinstance(model.model[15], BCRAUp) is flags[1]
    assert isinstance(model.model[32], CSTDDetect) is flags[2]
    assert list(model.model[32].f) == [23, 27, 31]
    assert torch.allclose(model.model[32].stride.cpu(), torch.tensor([8.0, 16.0, 32.0]))


@pytest.mark.parametrize("scale", ("n", "s", "l", "x"))
def test_sbt_n_s_l_x_and_fuse_dynamic_forward(scale):
    model = YOLO(str(MODEL_DIR / f"yolov13{scale}-sbt.yaml")).model.eval()
    assert len(model.model) == 33 and isinstance(model.model[4], SCPGDSC3k2)
    assert isinstance(model.model[15], BCRAUp) and isinstance(model.model[32], CSTDDetect)
    assert list(model.model[32].f) == [23, 27, 31]
    if scale == "n":
        with torch.no_grad():
            assert torch.isfinite(model(torch.randn(1, 3, 128, 128))[0]).all()
            assert torch.isfinite(model(torch.randn(1, 3, 160, 160))[0]).all()
        model.fuse(verbose=False)
        with torch.no_grad():
            assert torch.isfinite(model(torch.randn(1, 3, 128, 128))[0]).all()


@pytest.mark.parametrize("with_target", (False, True))
def test_sbt_loss_for_empty_and_normal_gt(with_target):
    model = YOLO(str(MODEL_DIR / "yolov13-sbt-t7-full.yaml")).model.train()
    model.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)
    batch = {
        "img": torch.randn(1, 3, 128, 128),
        "batch_idx": torch.tensor([0], dtype=torch.long) if with_target else torch.empty(0, dtype=torch.long),
        "cls": torch.tensor([[0.0]]) if with_target else torch.empty((0, 1)),
        "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]) if with_target else torch.empty((0, 4)),
    }
    loss, _ = model.loss(batch)
    assert torch.isfinite(loss)
    loss.backward()
