"""Regression tests for SEAL modules, parser wiring, loading and loss safety."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import torch

from ultralytics.nn.modules import (
    DegradationSeparatedReliabilityFilter,
    P3QualityDecoupler,
    QualityAlignedDecoupledDetect,
    SemanticAgreementReservoir,
)
from ultralytics.nn.tasks import DetectionModel, yaml_model_load
from ultralytics.utils import DEFAULT_CFG_DICT
from ultralytics.utils import IterableSimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "ultralytics/cfg/models/v13/seal_ablation"
CARM_A3 = ROOT / "ultralytics/cfg/models/v13/carm_ablation/yolov13-carm-a3-macr.yaml"


def build(path: Path) -> DetectionModel:
    return DetectionModel(cfg=yaml_model_load(path), ch=3, nc=None, verbose=False)


def assert_finite_gradients(module: torch.nn.Module) -> None:
    for parameter in module.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all(), parameter.shape


def test_zero_start_is_exact_identity() -> None:
    torch.manual_seed(7)
    shallow = torch.randn(2, 16, 12, 12)
    context = torch.randn(2, 32, 6, 6)
    dsrf = DegradationSeparatedReliabilityFilter(16)
    reservoir = SemanticAgreementReservoir(32, 16)
    decoupler = P3QualityDecoupler(16, 32)
    assert torch.equal(dsrf(shallow), shallow)
    assert torch.equal(reservoir(shallow, context), shallow)
    box, cls = decoupler(shallow, context)
    assert torch.equal(box, shallow)
    assert torch.equal(cls, shallow)


def test_module_gradients_are_finite_after_release() -> None:
    torch.manual_seed(8)
    shallow = torch.randn(2, 16, 12, 12, requires_grad=True)
    context = torch.randn(2, 32, 6, 6, requires_grad=True)
    dsrf = DegradationSeparatedReliabilityFilter(16)
    reservoir = SemanticAgreementReservoir(32, 16)
    decoupler = P3QualityDecoupler(16, 32)
    dsrf.alpha_raw.data.fill_(1.0)
    reservoir.alpha_raw.data.fill_(1.0)
    decoupler.box_alpha_raw.data.fill_(1.0)
    decoupler.cls_alpha_raw.data.fill_(1.0)
    loss = dsrf(shallow).square().mean() + reservoir(shallow, context).square().mean()
    box, cls = decoupler(shallow, context)
    loss = loss + box.square().mean() + cls.square().mean()
    loss.backward()
    assert torch.isfinite(shallow.grad).all()
    assert_finite_gradients(dsrf)
    assert_finite_gradients(reservoir)
    assert_finite_gradients(decoupler)


def test_all_seal_ablations_build_and_keep_carm_topology() -> None:
    expected = {
        "s0-carm-a3": ("DSC3k2", "DSC3k2", "Detect"),
        "s1-dsrf": ("DSRFDSC3k2", "DSC3k2", "Detect"),
        "s2-sarb": ("DSC3k2", "SARBDSC3k2", "Detect"),
        "s3-qad": ("DSC3k2", "DSC3k2", "QualityAlignedDecoupledDetect"),
        "s4-dsrf-sarb": ("DSRFDSC3k2", "SARBDSC3k2", "Detect"),
        "s5-dsrf-qad": ("DSRFDSC3k2", "DSC3k2", "QualityAlignedDecoupledDetect"),
        "s6-sarb-qad": ("DSC3k2", "SARBDSC3k2", "QualityAlignedDecoupledDetect"),
        "s7-full": ("DSRFDSC3k2", "SARBDSC3k2", "QualityAlignedDecoupledDetect"),
    }
    for stage, expected_types in expected.items():
        model = build(CONFIG_DIR / f"yolov13-seal-{stage}.yaml")
        assert len(model.model) == 33
        assert tuple(model.stride.tolist()) == (8.0, 16.0, 32.0)
        assert tuple(layer.__class__.__name__ for layer in (model.model[2], model.model[17], model.model[32])) == expected_types


def test_carm_a3_weights_transfer_with_only_new_seal_keys_missing() -> None:
    baseline = build(CARM_A3)
    seal = build(CONFIG_DIR / "yolov13-seal-s7-full.yaml")
    baseline_state = baseline.state_dict()
    missing = [key for key in seal.state_dict() if key not in baseline_state]
    allowed = ("model.2.reliability_filter.", "model.17.reservoir.", "model.32.p3_decoupler.")
    assert missing
    assert all(key.startswith(allowed) for key in missing), missing
    compatible = {key: value for key, value in baseline_state.items() if key in seal.state_dict() and value.shape == seal.state_dict()[key].shape}
    assert len(compatible) == len(baseline_state)
    seal.load_state_dict(compatible, strict=False)


def test_real_and_empty_gt_losses_are_finite_before_backward() -> None:
    model = build(CONFIG_DIR / "yolov13-seal-s7-full.yaml")
    model.args = IterableSimpleNamespace(**DEFAULT_CFG_DICT)
    model.train()
    for batch_idx, cls, bboxes in (
        (
            torch.tensor([0], dtype=torch.long),
            torch.tensor([[0.0]]),
            torch.tensor([[0.5, 0.5, 0.25, 0.25]]),
        ),
        (torch.empty(0, dtype=torch.long), torch.empty(0, 1), torch.empty(0, 4)),
    ):
        model.zero_grad(set_to_none=True)
        batch = {
            "img": torch.randn(1, 3, 128, 128),
            "batch_idx": batch_idx,
            "cls": cls,
            "bboxes": bboxes,
        }
        loss, _ = model.loss(batch)
        assert torch.isfinite(loss)
        loss.backward()
        assert_finite_gradients(model)


def test_quality_aligned_detect_forward_and_fuse() -> None:
    model = build(CONFIG_DIR / "yolov13-seal-s7-full.yaml")
    assert isinstance(model.model[-1], QualityAlignedDecoupledDetect)
    model.eval()
    with torch.no_grad():
        output = model(torch.zeros(1, 3, 128, 128))
    prediction = output[0] if isinstance(output, tuple) else output
    assert torch.isfinite(prediction).all()
    fused = deepcopy(model).fuse().eval()
    with torch.no_grad():
        fused_output = fused(torch.zeros(1, 3, 128, 128))
    fused_prediction = fused_output[0] if isinstance(fused_output, tuple) else fused_output
    assert torch.isfinite(fused_prediction).all()
