"""Contract tests for GMR's SCGP, MCAS and gradient-isolated GIMR head."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from ultralytics import YOLO
from ultralytics.nn.modules import Detect, GIMRDetect, GradientIsolatedMicroReconciler, MCASUp, SCPGDSC3k2
from ultralytics.nn.modules.block import DSC3k2

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "ultralytics" / "cfg" / "models" / "v13"


def test_scgp_preserves_rng_and_initial_main_path():
    torch.manual_seed(123)
    base = DSC3k2(c1=64, c2=128, n=1, dsc3k=False, e=0.25).eval()
    base_state = torch.random.get_rng_state().clone()
    torch.manual_seed(123)
    scgp = SCPGDSC3k2(c_shallow=64, c_p3=64, c2=128, n=1, dsc3k=False, e=0.25).eval()
    assert torch.equal(base_state, torch.random.get_rng_state())
    shallow, p3 = torch.randn(1, 64, 32, 32), torch.randn(1, 64, 16, 16)
    with torch.no_grad():
        assert torch.equal(scgp([shallow, p3]), base(p3))


def test_gmr_worker_uses_project_ultralytics_from_outside_project(tmp_path):
    worker = ROOT / "tools" / "train_gmr_worker.py"
    result = subprocess.run(
        [sys.executable, str(worker), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_gimr_flat_background_has_zero_gate_and_correction():
    module = GradientIsolatedMicroReconciler(32, 64).eval()
    p3, p4 = torch.zeros(2, 32, 40, 40), torch.zeros(2, 64, 20, 20)
    refined, correction, gate, _, detail_support, _ = module.compute_components(p3, p4)
    assert torch.equal(refined, p3)
    assert torch.equal(correction, torch.zeros_like(correction))
    assert torch.equal(detail_support, torch.zeros_like(detail_support))
    assert torch.equal(gate, torch.zeros_like(gate))


def test_gimr_initial_detect_towers_and_rng_match_exactly():
    torch.manual_seed(123)
    base = Detect(nc=4, ch=(64, 128, 256))
    base_state = torch.random.get_rng_state().clone()
    base_towers = {name: parameter.detach().clone() for name, parameter in base.named_parameters() if name.startswith(("cv2.", "cv3."))}
    torch.manual_seed(123)
    head = GIMRDetect(nc=4, ch=(64, 128, 256)).train()
    assert torch.equal(base_state, torch.random.get_rng_state())
    head_towers = {name: parameter.detach().clone() for name, parameter in head.named_parameters() if name.startswith(("cv2.", "cv3."))}
    assert base_towers.keys() == head_towers.keys()
    assert all(torch.equal(base_towers[name], head_towers[name]) for name in base_towers)
    features = [torch.randn(2, 64, 80, 80), torch.randn(2, 128, 40, 40), torch.randn(2, 256, 20, 20)]
    outputs = head(features)
    expected = [torch.cat((head.cv2[index](features[index]), head.cv3[index](features[index])), dim=1) for index in range(3)]
    assert all(torch.equal(actual, reference) for actual, reference in zip(outputs, expected))


def test_gimr_activation_changes_only_p3_classification_channels():
    head = GIMRDetect(nc=4, ch=(32, 64, 128)).train()
    with torch.no_grad():
        head.p3_reconciler.out_proj.weight.normal_(mean=0.0, std=0.01)
    p3, p4, p5 = torch.randn(2, 32, 32, 32), torch.randn(2, 64, 16, 16), torch.randn(2, 128, 8, 8)
    outputs = head([p3, p4, p5])
    assert torch.equal(outputs[0][:, : 4 * head.reg_max], head.cv2[0](p3))
    assert torch.equal(outputs[1], torch.cat((head.cv2[1](p4), head.cv3[1](p4)), dim=1))
    assert torch.equal(outputs[2], torch.cat((head.cv2[2](p5), head.cv3[2](p5)), dim=1))
    assert not torch.equal(outputs[0][:, 4 * head.reg_max :], head.cv3[0](p3))


def test_gimr_auxiliary_branch_is_gradient_isolated_and_budgeted():
    module = GradientIsolatedMicroReconciler(32, 64, max_ratio=0.04, detach_context=True).train()
    with torch.no_grad():
        module.out_proj.weight.normal_(mean=0.0, std=1.0)
    p3 = torch.randn(2, 32, 32, 32, requires_grad=True)
    p4 = torch.randn(2, 64, 16, 16, requires_grad=True)
    _, correction, _, _, _, _ = module.compute_components(p3, p4)
    correction.square().mean().backward()
    assert p3.grad is None and p4.grad is None
    base_energy = p3.detach().float().square().mean(dim=(2, 3), keepdim=True).sqrt()
    correction_energy = correction.detach().float().square().mean(dim=(2, 3), keepdim=True).sqrt()
    assert (correction_energy <= 0.04 * base_energy + 1e-5).all()


def test_gimr_two_step_gradient_activation():
    head = GIMRDetect(nc=4, ch=(32, 64, 128)).train()
    features = [torch.randn(2, 32, 32, 32), torch.randn(2, 64, 16, 16), torch.randn(2, 128, 8, 8)]
    optimizer = torch.optim.SGD(head.parameters(), lr=0.1)
    sum(output.square().mean() for output in head(features)).backward()
    assert head.p3_reconciler.out_proj.weight.grad is not None
    assert head.p3_reconciler.out_proj.weight.grad.abs().sum() > 0
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    sum(output.square().mean() for output in head(features)).backward()
    assert head.p3_reconciler.p3_proj.conv.weight.grad.abs().sum() > 0
    assert head.p3_reconciler.p4_proj.conv.weight.grad.abs().sum() > 0
    assert head.p3_reconciler.fusion[0].conv.weight.grad.abs().sum() > 0
    assert head.p3_reconciler.compatibility[0].weight.grad.abs().sum() > 0


@pytest.mark.parametrize(
    "stage,flags",
    (
        ("r0-baseline", (False, False, False)), ("r1-scgp", (True, False, False)),
        ("r2-mcas", (False, True, False)), ("r3-gimr", (False, False, True)),
        ("r4-scgp-mcas", (True, True, False)), ("r5-scgp-gimr", (True, False, True)),
        ("r6-mcas-gimr", (False, True, True)), ("r7-full", (True, True, True)),
    ),
)
def test_gmr_full_factorial_parser(stage, flags):
    model = YOLO(str(MODEL_DIR / f"yolov13-gmr-{stage}.yaml")).model
    assert len(model.model) == 33
    assert isinstance(model.model[4], SCPGDSC3k2) is flags[0]
    assert isinstance(model.model[15], MCASUp) is flags[1]
    assert isinstance(model.model[32], GIMRDetect) is flags[2]
    assert list(model.model[32].f) == [23, 27, 31]
    assert torch.allclose(model.model[32].stride.cpu(), torch.tensor([8.0, 16.0, 32.0]))


@pytest.mark.parametrize("scale", ("n", "s", "l", "x"))
def test_gmr_scales_dynamic_shape_fuse_eval_and_export(scale):
    model = YOLO(str(MODEL_DIR / f"yolov13{scale}-gmr.yaml")).model.eval()
    assert isinstance(model.model[4], SCPGDSC3k2) and isinstance(model.model[15], MCASUp) and isinstance(model.model[32], GIMRDetect)
    if scale == "n":
        with torch.no_grad():
            assert torch.isfinite(model(torch.randn(1, 3, 128, 128))[0]).all()
            assert torch.isfinite(model(torch.randn(1, 3, 160, 160))[0]).all()
        model.fuse(verbose=False)
        with torch.no_grad():
            assert torch.isfinite(model(torch.randn(1, 3, 128, 128))[0]).all()
        head = GIMRDetect(nc=4, ch=(32, 64, 128)).eval()
        head.stride = torch.tensor([8.0, 16.0, 32.0])
        features = [torch.randn(1, 32, 32, 32), torch.randn(1, 64, 16, 16), torch.randn(1, 128, 8, 8)]
        with torch.no_grad():
            prediction, raw = head(features)
        assert prediction.ndim == 3 and len(raw) == 3
        head.export = True
        with torch.no_grad():
            assert head(features).ndim == 3


def test_gmr_loss_for_empty_and_normal_gt():
    model = YOLO(str(MODEL_DIR / "yolov13-gmr-r7-full.yaml")).model.train()
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
