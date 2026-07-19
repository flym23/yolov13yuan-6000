"""Focused regression tests for the reviewed DGDR-YOLOv13 structural changes."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ultralytics import YOLO
from ultralytics.nn.modules import CAGDSC3k2, ContourGuidedAdaptiveGeometry, DAPD, SBRHDetect


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "ultralytics" / "cfg" / "models" / "v13"


def test_dapd_identity_odd_shape_phase_order_and_gradient():
    module = DAPD(32, 32, k=3, s=2, p=1, g=4, max_gain=0.10).train()
    x = torch.randn(2, 32, 81, 79, requires_grad=True)
    y = module(x)
    base = module.act(module.bn(module.conv(x)))
    assert y.shape == (2, 32, 41, 40)
    assert torch.allclose(y, base, atol=1e-6, rtol=1e-6)
    residual = module._phase_residual(module._pad_even(x.detach())).view(2, 32, 4, 41, 40)
    assert torch.allclose(residual.sum(dim=2), torch.zeros_like(residual[:, :, 0]), atol=1e-5, rtol=1e-5)
    y.mean().backward()
    assert module.alpha_raw.grad is not None and torch.isfinite(module.alpha_raw.grad).all()
    assert module.alpha_raw.grad.abs().sum() > 0


def test_cagb_identity_geometry_bounds_and_gradient():
    module = ContourGuidedAdaptiveGeometry(32, samples=5, max_offset=2.0, max_gain=0.08).train()
    x = torch.randn(2, 32, 40, 40, requires_grad=True)
    y = module(x)
    assert y.shape == x.shape
    assert torch.allclose(y, x, atol=1e-6, rtol=1e-6)
    y.mean().backward()
    assert module.alpha_raw.grad is not None and module.alpha_raw.grad.abs().sum() > 0
    module.eval()
    with torch.no_grad():
        offsets, weights = module._predict_geometry(torch.randn(2, 32, 19, 21))
    assert offsets.shape == (2, 5, 2, 19, 21)
    assert offsets.abs().max() <= 2.0 + 1e-6
    assert torch.allclose(weights.sum(dim=1), torch.ones_like(weights[:, 0]), atol=1e-6, rtol=1e-6)


def test_sbrh_zero_start_gradient_eval_and_export_fallback():
    head = SBRHDetect(nc=4, refine_ratio=1.0, ch=(64, 128, 256)).train()
    features = [torch.randn(2, 64, 40, 40), torch.randn(2, 128, 20, 20), torch.randn(2, 256, 10, 10)]
    outputs = head(features)
    assert [tuple(output.shape) for output in outputs] == [(2, 68, 40, 40), (2, 68, 20, 20), (2, 68, 10, 10)]
    for index, feature in enumerate(features):
        reg = head.cv2[index][1](head.cv2[index][0](feature))
        assert torch.allclose(outputs[index][:, :64], head.cv2[index][2](reg), atol=1e-6, rtol=1e-6)
    sum(output.mean() for output in outputs).backward()
    assert head.side_alpha_raw.grad is not None and head.side_alpha_raw.grad.abs().sum() > 0
    head.eval()
    prediction, raw = head([feature[:1] for feature in features])
    assert prediction.shape[1] == 8 and len(raw) == 3
    head.export, head.format = True, "tflite"
    exported = head([feature[:1] for feature in features])
    assert exported.shape[1] == 8


@pytest.mark.parametrize(
    ("name", "modules"),
    [
        ("g1-dapd", (DAPD, None, None)),
        ("g2-cagb", (None, CAGDSC3k2, None)),
        ("g3-sbrh", (None, None, SBRHDetect)),
        ("g7-full", (DAPD, CAGDSC3k2, SBRHDetect)),
    ],
)
def test_dgdr_ablation_builds(name, modules):
    wrapper = YOLO(str(MODEL_DIR / f"yolov13n-dgdr-{name}.yaml"))
    layers = wrapper.model.model
    assert len(layers) == 33 and list(layers[32].f) == [23, 27, 31]
    for index, expected in zip((3, 4, 32), modules):
        if expected is not None:
            assert isinstance(layers[index], expected)


@pytest.mark.parametrize(
    ("scale", "dapd_channels", "cagb_channels", "detect_channels", "repeats", "dsc3k"),
    [
        ("n", (64, 64), (64, 128), [64, 128, 256], 1, False),
        ("s", (128, 128), (128, 256), [128, 256, 512], 1, False),
        ("l", (256, 256), (256, 512), [256, 512, 512], 2, True),
        ("x", (384, 384), (384, 512), [384, 512, 512], 2, True),
    ],
)
def test_full_dgdr_scale_parsing(scale, dapd_channels, cagb_channels, detect_channels, repeats, dsc3k):
    wrapper = YOLO(str(MODEL_DIR / f"yolov13{scale}-dgdr.yaml"))
    layers = wrapper.model.model
    dapd, cagb, detect = layers[3], layers[4], layers[32]
    assert isinstance(dapd, DAPD) and isinstance(cagb, CAGDSC3k2) and isinstance(detect, SBRHDetect)
    assert (dapd.conv.in_channels, dapd.conv.out_channels) == dapd_channels
    assert (cagb.cv1.conv.in_channels, cagb.cv2.conv.out_channels) == cagb_channels
    assert len(cagb.m) == repeats and cagb.dsc3k_enabled is dsc3k
    assert [tower[0].conv.in_channels for tower in detect.cv2] == detect_channels
    assert torch.allclose(detect.stride.cpu(), torch.tensor([8.0, 16.0, 32.0]))


def test_g7_fuse_keeps_dapd_branch_and_runs_forward():
    wrapper = YOLO(str(MODEL_DIR / "yolov13n-dgdr-g7-full.yaml"))
    model = wrapper.model.eval()
    model.fuse(verbose=False)
    assert isinstance(model.model[3], DAPD)
    with torch.no_grad():
        prediction = model(torch.randn(1, 3, 128, 128))[0]
    assert torch.isfinite(prediction).all()


@pytest.mark.parametrize("with_target", (False, True))
def test_g7_uses_original_detection_loss_for_empty_and_normal_gt(with_target):
    model = YOLO(str(MODEL_DIR / "yolov13n-dgdr-g7-full.yaml")).model.train()
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
