"""Identity, residual-budget, and model-contract tests for DCRQ-YOLOv13."""

from pathlib import Path

import pytest
import torch

from ultralytics import YOLO
from ultralytics.nn.modules.dcrq import DAWRB, RCFConcat, RecallPreservingQualityCalibrator
from ultralytics.nn.modules.head import RPQUDQDetect


ROOT = Path(__file__).resolve().parents[1]
YAML_DIR = ROOT / "ultralytics" / "cfg" / "models" / "v13"


def _rms_ratio(base, residual):
    return (residual.float().square().mean((2, 3)).sqrt() / base.float().square().mean((2, 3)).sqrt().clamp_min(1e-12)).max()


def _finite_gradients(module):
    gradients = [parameter.grad for parameter in module.parameters() if parameter.requires_grad and parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_dawrb_identity_even_shape():
    module, x = DAWRB(16), torch.randn(2, 16, 80, 80)
    output = module(x)
    assert output.shape == x.shape
    assert torch.isfinite(output).all()
    assert torch.equal(output, x)


def test_dawrb_identity_odd_shape():
    module, x = DAWRB(16), torch.randn(2, 16, 81, 79)
    output = module(x)
    assert output.shape == x.shape
    assert torch.isfinite(output).all()
    assert torch.equal(output, x)


def test_dawrb_budget_nonzero():
    torch.manual_seed(0)
    module, x = DAWRB(16), torch.randn(2, 16, 31, 29)
    nn_weight = module.band_projection.weight
    torch.nn.init.normal_(nn_weight, mean=0.0, std=0.02)
    residual = module(x) - x
    assert float(_rms_ratio(x, residual)) <= 0.05 + 2e-5


def test_dawrb_backward_finite():
    torch.manual_seed(1)
    module, x = DAWRB(16), torch.randn(2, 16, 31, 29, requires_grad=True)
    torch.nn.init.normal_(module.band_projection.weight, mean=0.0, std=0.02)
    module(x).float().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    _finite_gradients(module)


def test_rcfconcat_identity():
    module = RCFConcat(16, 24)
    deep, lateral = torch.randn(2, 16, 40, 40), torch.randn(2, 24, 40, 40)
    output = module([deep, lateral])
    assert output.shape == (2, 40, 40, 40)
    assert torch.equal(output, torch.cat((deep, lateral), dim=1))


def test_rcfconcat_lateral_unchanged():
    module = RCFConcat(16, 24)
    deep, lateral = torch.randn(2, 16, 40, 40), torch.randn(2, 24, 40, 40)
    assert torch.equal(module([deep, lateral])[:, 16:], lateral)


def test_rcfconcat_budget_nonzero():
    torch.manual_seed(2)
    module = RCFConcat(16, 24)
    deep, lateral = torch.randn(2, 16, 40, 40), torch.randn(2, 24, 40, 40)
    torch.nn.init.normal_(module.out_projection.weight, mean=0.0, std=0.02)
    output = module([deep, lateral])
    assert float(_rms_ratio(deep, output[:, :16] - deep)) <= 0.04 + 2e-5
    assert torch.equal(output[:, 16:], lateral)


def test_rcfconcat_backward_finite():
    torch.manual_seed(3)
    module = RCFConcat(16, 24)
    deep, lateral = torch.randn(2, 16, 40, 40, requires_grad=True), torch.randn(2, 24, 40, 40, requires_grad=True)
    torch.nn.init.normal_(module.out_projection.weight, mean=0.0, std=0.02)
    output = module([deep, lateral])
    output.float().mean().backward()
    assert torch.isfinite(deep.grad).all() and torch.isfinite(lateral.grad).all()
    _finite_gradients(module)


def test_rpqc_monotonic_quality():
    calibrator = RecallPreservingQualityCalibrator()
    cls = torch.full((1, 1, 1, 5), 0.4)
    quality = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0]).view(1, 1, 1, 5)
    values = calibrator(cls, quality, 0).flatten()
    assert torch.all(values[1:] >= values[:-1])


def test_rpqc_low_quality_weak_suppression():
    calibrator = RecallPreservingQualityCalibrator()
    cls, quality = torch.ones(1, 1, 1, 1), torch.zeros(1, 1, 1, 1)
    assert calibrator(cls, quality, 0).item() == pytest.approx(0.96875)


def test_rpqc_high_quality_boost():
    calibrator = RecallPreservingQualityCalibrator()
    cls, quality = torch.full((1, 1, 1, 1), 0.4), torch.ones(1, 1, 1, 1)
    assert calibrator(cls, quality, 0).item() == pytest.approx(0.4 * 1.125)


@pytest.mark.parametrize("stage", ("b2", "b3", "b4"))
def test_dcrq_model_build(stage):
    model = YOLO(str(YAML_DIR / f"yolov13n-dcrq-{stage}.yaml"))
    assert isinstance(model.model.model[-1], RPQUDQDetect)
    assert len(model.model.model[-1].stride) == 3


def test_b4_detect_strides_and_forward_contract():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = YOLO(str(YAML_DIR / "yolov13n-dcrq-b4.yaml")).model.to(device).eval()
    assert model.model[-1].stride.tolist() == pytest.approx([8.0, 16.0, 32.0])
    with torch.inference_mode():
        prediction, raw = model(torch.randn(1, 3, 640, 640, device=device))
    assert prediction.shape[1] == 8  # xywh plus four classes
    assert [feature.shape[-2:] for feature in raw] == [(80, 80), (40, 40), (20, 20)]
