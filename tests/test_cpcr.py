"""Contract tests for CPCR-YOLOv13 detached class-prototype recovery."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from ultralytics import YOLO
from ultralytics.nn.modules import CPCRDetect, ClassPrototypeComplementaryRecovery, GradientIsolatedShallowEncoder, SCPGDSC3k2

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "ultralytics" / "cfg" / "models" / "v13"


def test_gradient_isolated_encoder_shape_and_detach():
    module = GradientIsolatedShallowEncoder(32, 64, 64, reduction=4, detach_guidance=True).train()
    shallow, p3 = torch.randn(2, 32, 81, 79, requires_grad=True), torch.randn(2, 64, 41, 40, requires_grad=True)
    auxiliary, prior = module(shallow, p3)
    assert auxiliary.shape == (2, 64, 41, 40) and prior.shape == (2, 1, 41, 40)
    assert torch.isfinite(auxiliary).all() and torch.isfinite(prior).all() and 0.0 <= prior.min() and prior.max() <= 1.0
    auxiliary.mean().backward()
    assert shallow.grad is None and p3.grad is None and module.fuse[0].conv.weight.grad is not None


def test_prototype_weight_is_detached_and_delta_is_bounded():
    module = ClassPrototypeComplementaryRecovery(32, 64, 64, nc=4, reg_max=16).train()
    classifier = nn.Conv2d(64, 4, 1, bias=True)
    output, diagnostics = module(torch.randn(2, 32, 80, 80), torch.randn(2, 64, 40, 40), torch.randn(2, 64, 40, 40),
                                 torch.randn(2, 4, 40, 40, requires_grad=True), torch.randn(2, 64, 40, 40), classifier)
    output.mean().backward()
    delta = diagnostics["delta"]
    assert classifier.weight.grad is None and module.candidate_bias.grad is not None and module.gain_raw.grad is not None
    assert module.class_gate[-1].weight.grad is not None and torch.isfinite(delta).all()
    assert delta.min() >= 0.0 and delta.max() <= module.max_delta + 1e-6


def test_localization_guard_bounds():
    module = ClassPrototypeComplementaryRecovery(32, 64, 64, nc=4, reg_max=16, loc_floor=0.50).eval()
    guard = module._localization_guard(torch.randn(2, 64, 40, 40))
    assert guard.shape == (2, 1, 40, 40) and torch.isfinite(guard).all() and guard.min() >= 0.50 and guard.max() <= 1.0


def test_cpcr_detect_three_scale_and_zero_delta_equivalence():
    head = CPCRDetect(nc=4, c_shallow=64, ch=(64, 128, 256)).train()
    outputs = head([torch.randn(2, 64, 160, 160), torch.randn(2, 64, 80, 80), torch.randn(2, 128, 40, 40), torch.randn(2, 256, 20, 20)])
    assert [tuple(output.shape) for output in outputs] == [(2, 68, 80, 80), (2, 68, 40, 40), (2, 68, 20, 20)]
    zero = CPCRDetect(nc=4, max_delta=0.0, c_shallow=32, ch=(32, 64, 128)).train()
    p2, p3, p4, p5 = torch.randn(1, 32, 64, 64), torch.randn(1, 32, 32, 32), torch.randn(1, 64, 16, 16), torch.randn(1, 128, 8, 8)
    for index, feature in enumerate((p3, p4, p5)):
        expected = torch.cat((zero.cv2[index](feature), zero.cv3[index](feature)), dim=1)
        assert torch.allclose(zero([p2, p3, p4, p5])[index], expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("stage,flags", (("l1-prototype", (False, False, False)), ("l2-spatial", (True, False, False)), ("l3-class", (True, True, False)), ("l4-full", (True, True, True)), ("l5-h3-full", (True, True, True))))
def test_cpcr_yaml_builds(stage, flags):
    model = YOLO(str(MODEL_DIR / f"yolov13n-cpcr-{stage}.yaml")).model
    head = model.model[32]
    assert len(model.model) == 33 and list(head.f) == [2, 23, 27, 31] and isinstance(head, CPCRDetect) and head.nl == 3
    assert (head.recovery.use_spatial_prior, head.recovery.use_class_gate, head.recovery.use_loc_guard) == flags
    assert isinstance(model.model[4], SCPGDSC3k2) is (stage == "l5-h3-full")
    assert torch.allclose(head.stride.cpu(), torch.tensor([8.0, 16.0, 32.0]))


@pytest.mark.parametrize("scale,channels", (("n", (64, [64, 128, 256])), ("s", (128, [128, 256, 512])), ("l", (256, [256, 512, 512])), ("x", (384, [384, 512, 512]))))
def test_cpcr_scale_parsing(scale, channels):
    model = YOLO(str(MODEL_DIR / f"yolov13{scale}-cpcr.yaml")).model
    head = model.model[32]
    assert isinstance(head, CPCRDetect) and head.c_shallow == channels[0] and [tower[0].conv.in_channels for tower in head.cv2] == channels[1]


@pytest.mark.parametrize("with_target", (False, True))
def test_cpcr_uses_original_detection_loss_for_empty_and_normal_gt(with_target):
    model = YOLO(str(MODEL_DIR / "yolov13n-cpcr-l4-full.yaml")).model.train()
    model.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)
    batch = {"img": torch.randn(1, 3, 128, 128), "batch_idx": torch.tensor([0], dtype=torch.long) if with_target else torch.empty(0, dtype=torch.long),
             "cls": torch.tensor([[0.0]]) if with_target else torch.empty((0, 1)), "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]) if with_target else torch.empty((0, 4))}
    loss, _ = model.loss(batch)
    assert torch.isfinite(loss)
    loss.backward()
