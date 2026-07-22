"""Contract tests for RAMP-YOLOv13 P2-guided P3 classification reactivation."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ultralytics import YOLO
from ultralytics.nn.modules import AmbiguityReactivationGate, HighResolutionEvidenceEncoder, P2RecallReactivation, RAMPDetect, SCPGDSC3k2

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "ultralytics" / "cfg" / "models" / "v13"


def test_high_resolution_encoder_odd_shape_and_bounds():
    module = HighResolutionEvidenceEncoder(c_shallow=64, c_p3=64, reduction=4).train()
    shallow = torch.randn(2, 64, 161, 159, requires_grad=True)
    evidence, prior = module(shallow, (81, 80))
    assert evidence.shape == (2, module.evidence_channels, 81, 80) and prior.shape == (2, 1, 81, 80)
    assert torch.isfinite(evidence).all() and torch.isfinite(prior).all() and 0.0 <= prior.min() and prior.max() <= 1.0


def test_ambiguity_gate_shape_bounds_and_gradient_contract():
    encoder = HighResolutionEvidenceEncoder(c_shallow=32, c_p3=64)
    gate = AmbiguityReactivationGate(evidence_channels=encoder.evidence_channels, c_p3=64)
    shallow, p3 = torch.randn(2, 32, 80, 80), torch.randn(2, 64, 40, 40, requires_grad=True)
    evidence, prior = encoder(shallow, p3.shape[-2:])
    ambiguity = gate(evidence, prior, p3)
    assert ambiguity.shape == (2, 1, 40, 40) and torch.isfinite(ambiguity).all()
    assert 0.0 <= ambiguity.min() and ambiguity.max() <= 1.0
    ambiguity.mean().backward()
    # Gradients may flow through learned compatibility; deterministic prior/confidence conditions are detached in the module.
    assert p3.grad is not None and torch.isfinite(p3.grad).all()


def test_reactivation_is_near_identity_non_suppressive_and_trainable():
    module = P2RecallReactivation(c_shallow=32, c_p3=64, max_gain=0.10, gain_init=-4.0).train()
    shallow = torch.randn(2, 32, 80, 80, requires_grad=True)
    p3 = torch.randn(2, 64, 40, 40, requires_grad=True)
    output, gate = module(shallow, p3)
    assert output.shape == p3.shape and gate.shape == (2, 1, 40, 40)
    assert torch.isfinite(output).all() and torch.isfinite(gate).all()
    assert (output.abs() + 1e-7 >= p3.abs()).all() and (output - p3).abs().max() < 5e-3
    (output.square().mean() + 0.01 * output.mean()).backward()
    assert module.gain_raw.grad is not None and module.gain_raw.grad.abs().sum() > 0
    assert module.channel_gate.weight.grad is not None and torch.isfinite(module.channel_gate.weight.grad).all()


def test_ramp_detect_three_scale_contract_and_zero_gain_equivalence():
    head = RAMPDetect(nc=4, c_shallow=64, ch=(64, 128, 256)).train()
    features = [torch.randn(2, 64, 160, 160), torch.randn(2, 64, 80, 80), torch.randn(2, 128, 40, 40), torch.randn(2, 256, 20, 20)]
    outputs = head(features)
    assert [tuple(output.shape) for output in outputs] == [(2, 68, 80, 80), (2, 68, 40, 40), (2, 68, 20, 20)]

    zero = RAMPDetect(nc=4, max_gain=0.0, c_shallow=32, ch=(32, 64, 128)).train()
    p2, p3, p4, p5 = torch.randn(1, 32, 64, 64), torch.randn(1, 32, 32, 32), torch.randn(1, 64, 16, 16), torch.randn(1, 128, 8, 8)
    outputs = zero([p2, p3, p4, p5])
    for index, feature in enumerate((p3, p4, p5)):
        expected = torch.cat((zero.cv2[index](feature), zero.cv3[index](feature)), dim=1)
        assert torch.allclose(outputs[index], expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(
    ("stage", "layer4", "ambiguity", "channel"),
    (("k1-baseline", "DSC3k2", True, True), ("k2-prior", SCPGDSC3k2, False, False), ("k3-ambiguity", SCPGDSC3k2, True, False), ("k4-full", SCPGDSC3k2, True, True)),
)
def test_ramp_yaml_builds(stage, layer4, ambiguity, channel):
    model = YOLO(str(MODEL_DIR / f"yolov13n-ramp-{stage}.yaml")).model
    head = model.model[32]
    assert len(model.model) == 33 and list(head.f) == [2, 23, 27, 31] and isinstance(head, RAMPDetect) and head.nl == 3
    assert model.model[4].__class__.__name__ == layer4 if isinstance(layer4, str) else isinstance(model.model[4], layer4)
    assert head.p3_reactivation.use_ambiguity is ambiguity and head.p3_reactivation.use_channel is channel
    assert torch.allclose(head.stride.cpu(), torch.tensor([8.0, 16.0, 32.0]))


@pytest.mark.parametrize(
    ("scale", "channels", "repeats", "dsc3k"),
    (("n", (64, 64, [64, 128, 256]), 1, False), ("s", (128, 128, [128, 256, 512]), 1, False), ("l", (256, 256, [256, 512, 512]), 2, True), ("x", (384, 384, [384, 512, 512]), 2, True)),
)
def test_full_ramp_scale_parsing(scale, channels, repeats, dsc3k):
    model = YOLO(str(MODEL_DIR / f"yolov13{scale}-ramp.yaml")).model
    scpg, head = model.model[4], model.model[32]
    assert isinstance(scpg, SCPGDSC3k2) and isinstance(head, RAMPDetect)
    assert (scpg.c_shallow, scpg.c_p3) == channels[:2] and head.c_shallow == channels[0]
    assert len(scpg.m) == repeats and scpg.dsc3k_enabled is dsc3k
    assert [tower[0].conv.in_channels for tower in head.cv2] == channels[2] and head.nl == 3


@pytest.mark.parametrize("with_target", (False, True))
def test_ramp_uses_original_detection_loss_for_empty_and_normal_gt(with_target):
    model = YOLO(str(MODEL_DIR / "yolov13n-ramp-k4-full.yaml")).model.train()
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
