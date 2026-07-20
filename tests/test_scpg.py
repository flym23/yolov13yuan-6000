"""Regression tests for the reviewed SCPG-YOLOv13 H1--H5 structural changes."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ultralytics import YOLO
from ultralytics.nn.modules import (
    CenterPreservedPartialGeometry,
    P3DecoupledDetect,
    P3TaskAdapter,
    SCPGDSC3k2,
    ShallowEvidenceRouter,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "ultralytics" / "cfg" / "models" / "v13"


def test_ser_identity_odd_shape_and_gain_gradient():
    module = ShallowEvidenceRouter(32, 64, max_gain=0.08).train()
    shallow = torch.randn(2, 32, 81, 79, requires_grad=True)
    semantic = torch.randn(2, 64, 41, 40, requires_grad=True)
    output, micro = module(shallow, semantic)
    assert output.shape == semantic.shape and micro.shape == (2, 1, 41, 40)
    assert torch.allclose(output, semantic, atol=1e-6, rtol=1e-6)
    (output.mean() + micro.mean()).backward()
    assert module.alpha_raw.grad is not None and torch.isfinite(module.alpha_raw.grad).all()
    assert module.alpha_raw.grad.abs().sum() > 0


def test_cppg_identity_bounds_weight_normalization_and_gradient():
    module = CenterPreservedPartialGeometry(64, geom_ratio=0.5, samples=5).train()
    x = torch.randn(2, 64, 31, 29, requires_grad=True)
    micro = torch.rand(2, 1, 31, 29)
    with torch.no_grad():
        offsets, weights = module._predict_geometry(module.reduce(x.detach()), micro)
    assert offsets.shape == (2, 5, 2, 31, 29)
    assert offsets.abs().max() <= 1.5 + 1e-6
    assert torch.allclose(weights.sum(dim=1), torch.ones_like(weights[:, 0]), atol=1e-6, rtol=1e-6)
    assert (weights[:, 0] >= 0.5).all()
    output = module(x, micro)
    assert torch.allclose(output, x, atol=1e-6, rtol=1e-6)
    output.mean().backward()
    assert module.alpha_raw.grad is not None and module.alpha_raw.grad.abs().sum() > 0


def test_scpg_block_identity_is_original_dsc3k2_main_path():
    module = SCPGDSC3k2(c_shallow=64, c_p3=64, c2=128, n=1, dsc3k=False, e=0.25).train()
    p2, p3 = torch.randn(2, 64, 160, 160), torch.randn(2, 64, 80, 80)
    output = module([p2, p3])
    original = super(SCPGDSC3k2, module).forward(p3)
    assert output.shape == (2, 128, 80, 80)
    assert torch.allclose(output, original, atol=1e-6, rtol=1e-6)


def test_p3_adapter_and_detect_preserve_initial_detect_logits():
    adapter = P3TaskAdapter(64, max_gain=0.10).train()
    x = torch.randn(2, 64, 40, 40, requires_grad=True)
    reg, cls = adapter(x)
    assert torch.allclose(reg, x, atol=1e-6, rtol=1e-6)
    assert torch.allclose(cls, x, atol=1e-6, rtol=1e-6)
    (reg.mean() + cls.mean()).backward()
    assert adapter.alpha_raw.grad is not None and adapter.alpha_raw.grad.abs().sum() > 0

    head = P3DecoupledDetect(nc=4, max_gain=0.10, reduction=4, ch=(64, 128, 256)).train()
    features = [torch.randn(2, 64, 80, 80), torch.randn(2, 128, 40, 40), torch.randn(2, 256, 20, 20)]
    outputs = head(features)
    assert [tuple(item.shape) for item in outputs] == [(2, 68, 80, 80), (2, 68, 40, 40), (2, 68, 20, 20)]
    for index, feature in enumerate(features):
        expected = torch.cat((head.cv2[index](feature), head.cv3[index](feature)), dim=1)
        assert torch.allclose(outputs[index], expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(
    ("stage", "layer4", "head"),
    (
        ("h1-ser", SCPGDSC3k2, "Detect"),
        ("h2-cppg", SCPGDSC3k2, "Detect"),
        ("h3-ser-cppg", SCPGDSC3k2, "Detect"),
        ("h4-p3tda", "DSC3k2", P3DecoupledDetect),
        ("h5-full", SCPGDSC3k2, P3DecoupledDetect),
    ),
)
def test_scpg_h1_h5_yaml_builds(stage, layer4, head):
    model = YOLO(str(MODEL_DIR / f"yolov13n-scpg-{stage}.yaml")).model
    layers = model.model
    assert len(layers) == 33 and list(layers[32].f) == [23, 27, 31]
    assert layers[4].__class__.__name__ == layer4 if isinstance(layer4, str) else isinstance(layers[4], layer4)
    assert layers[32].__class__.__name__ == head if isinstance(head, str) else isinstance(layers[32], head)


@pytest.mark.parametrize(
    ("scale", "channels", "repeats", "dsc3k"),
    (
        ("n", (64, 64, 128, [64, 128, 256]), 1, False),
        ("s", (128, 128, 256, [128, 256, 512]), 1, False),
        ("l", (256, 256, 512, [256, 512, 512]), 2, True),
        ("x", (384, 384, 512, [384, 512, 512]), 2, True),
    ),
)
def test_full_scpg_scale_parsing(scale, channels, repeats, dsc3k):
    model = YOLO(str(MODEL_DIR / f"yolov13{scale}-scpg.yaml")).model
    scpg, detect = model.model[4], model.model[32]
    assert isinstance(scpg, SCPGDSC3k2) and isinstance(detect, P3DecoupledDetect)
    assert (scpg.c_shallow, scpg.c_p3, scpg.c2) == channels[:3]
    assert len(scpg.m) == repeats and scpg.dsc3k_enabled is dsc3k
    assert [tower[0].conv.in_channels for tower in detect.cv2] == channels[3]
    assert torch.allclose(detect.stride.cpu(), torch.tensor([8.0, 16.0, 32.0]))


@pytest.mark.parametrize("with_target", (False, True))
def test_scpg_uses_original_detection_loss_for_empty_and_normal_gt(with_target):
    model = YOLO(str(MODEL_DIR / "yolov13n-scpg-h5-full.yaml")).model.train()
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

