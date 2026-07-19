"""Focused regression tests for DURC-YOLOv13 modules and loss integration."""

from pathlib import Path

import pytest
import torch
import yaml

from tools.prepare_durc_dataset import build_durc_dataset, normalize_label_line
from ultralytics.nn.modules.block import DRSC, DRSCDSC3k2, DSC3k2
from ultralytics.nn.modules.head import Detect, HRCTDetect, NonUniformDFL, SUDLDetect
from ultralytics import YOLO
from ultralytics.utils.loss import (
    BboxLoss,
    DFLoss,
    SUDLBboxLoss,
    SUDLDFLoss,
    VarifocalLoss,
    v8DetectionLoss,
)


def test_durc_private_dataset_maps_one_based_labels_without_mutating_source(
    tmp_path: Path,
):
    source_root = tmp_path / "source"
    images = source_root / "images" / "train"
    labels = source_root / "labels" / "train"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    (images / "sample.jpg").touch()
    source_label = labels / "sample.txt"
    source_label.write_text("1 0.5 0.5 0.2 0.2\n4 0.4 0.4 0.3 0.3\n", encoding="utf-8")
    source_data = tmp_path / "source.yaml"
    source_data.write_text(
        yaml.safe_dump(
            {
                "path": str(source_root),
                "train": "images/train",
                "names": {0: "a", 1: "b", 2: "c", 3: "d"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    target_root = tmp_path / "durc_dataset"
    target_data = tmp_path / "durc.yaml"
    generated = build_durc_dataset(source_data, target_root, target_data)

    assert generated["train"] == "images/train"
    assert (target_root / "images" / "train").is_dir()
    assert (target_root / "images" / "train" / "sample.jpg").samefile(
        images / "sample.jpg"
    )
    assert (target_root / "labels" / "train" / "sample.txt").read_text(
        encoding="utf-8"
    ) == ("0 0.5 0.5 0.2 0.2\n3 0.4 0.4 0.3 0.3\n")
    assert source_label.read_text(encoding="utf-8").startswith("1 ")
    with pytest.raises(ValueError, match="outside"):
        normalize_label_line("0 0.5 0.5 0.2 0.2", source_label, 1)


def test_drsc_identity_shape_gradient_and_constant_boundary():
    torch.manual_seed(0)
    module = DRSC(16, reduction=4, max_gain=0.1, pool_kernel=5)
    x = torch.randn(2, 16, 17, 19, requires_grad=True)
    output = module(x)
    assert output.shape == x.shape
    assert torch.allclose(output, x, atol=1e-6, rtol=1e-6)
    output.square().mean().backward()
    assert module.alpha_raw.grad is not None
    assert torch.isfinite(module.alpha_raw.grad).all()

    constant = torch.full((2, 16, 11, 13), 3.25)
    residual = constant - module._lowpass(constant)
    assert torch.allclose(residual, torch.zeros_like(residual), atol=1e-6, rtol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_drsc_cuda_fp16_is_finite():
    module = DRSC(16).cuda().half()
    x = torch.randn(
        2, 16, 16, 16, device="cuda", dtype=torch.float16, requires_grad=True
    )
    output = module(x)
    assert torch.isfinite(output).all()
    output.float().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert (
        module.alpha_raw.grad is not None
        and torch.isfinite(module.alpha_raw.grad).all()
    )


def test_drscdsc3k2_shape_and_state_migration():
    torch.manual_seed(1)
    original = DSC3k2(16, 32, n=2, dsc3k=False, e=0.25)
    wrapped = DRSCDSC3k2(16, 32, n=2, dsc3k=False, e=0.25)
    incompatible = wrapped.load_state_dict(original.state_dict(), strict=False)
    assert not incompatible.unexpected_keys
    assert incompatible.missing_keys
    assert all(key.startswith("drsc.") for key in incompatible.missing_keys)
    output = wrapped(torch.randn(2, 16, 20, 20))
    assert output.shape == (2, 32, 20, 20)


def _copy_detect_parameters(source: Detect, target: Detect):
    target.cv2.load_state_dict(source.cv2.state_dict())
    target.cv3.load_state_dict(source.cv3.state_dict())
    target.dfl.load_state_dict(source.dfl.state_dict())


def test_hrct_detect_identity_roles_disabled_nodes_and_input_preservation():
    torch.manual_seed(2)
    channels = (16, 32, 64)
    original = Detect(nc=4, ch=channels).train()
    calibrated = HRCTDetect(nc=4, ch=channels).train()
    _copy_detect_parameters(original, calibrated)

    features = [
        torch.randn(2, 16, 20, 20),
        torch.randn(2, 32, 10, 10),
        torch.randn(2, 64, 5, 5),
    ]
    snapshots = [feature.clone() for feature in features]
    expected = original([feature.clone() for feature in features])
    actual = calibrated(features)
    assert all(torch.equal(before, after) for before, after in zip(snapshots, features))
    assert all(
        torch.allclose(a, b, atol=1e-6, rtol=1e-6) for a, b in zip(actual, expected)
    )
    assert calibrated.hrct_p3.role == "detail"
    assert calibrated.hrct_p4.role == "balanced"
    assert calibrated.hrct_p5.role == "semantic"

    partial = HRCTDetect(nc=4, p3_gain=0.1, p4_gain=0.0, p5_gain=0.0, ch=channels)
    assert partial.hrct_p3 is not None
    assert partial.hrct_p4 is None
    assert partial.hrct_p5 is None
    assert not any("hrct_p4" in key or "hrct_p5" in key for key in partial.state_dict())


def test_hrct_detect_training_and_inference_outputs_are_finite():
    head = HRCTDetect(nc=4, ch=(16, 32, 64))
    head.stride = torch.tensor([8.0, 16.0, 32.0])
    features = [
        torch.randn(2, 16, 20, 20),
        torch.randn(2, 32, 10, 10),
        torch.randn(2, 64, 5, 5),
    ]
    train_output = head.train()([feature.clone() for feature in features])
    assert len(train_output) == 3
    assert all(torch.isfinite(item).all() for item in train_output)
    inference, raw = head.eval()([feature.clone() for feature in features])
    assert torch.isfinite(inference).all()
    assert len(raw) == 3 and all(torch.isfinite(item).all() for item in raw)


def test_non_uniform_dfl_projection_shape_and_backward():
    uniform = NonUniformDFL(reg_max=16, gamma=1.0)
    assert torch.equal(uniform.project, torch.arange(16, dtype=torch.float32))

    non_uniform = NonUniformDFL(reg_max=16, gamma=1.5)
    assert non_uniform.project[0].item() == 0.0
    assert non_uniform.project[-1].item() == 15.0
    assert torch.all(non_uniform.project[1:] > non_uniform.project[:-1])
    logits = torch.randn(2, 64, 25, requires_grad=True)
    output = non_uniform(logits)
    assert output.shape == (2, 4, 25)
    output.mean().backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_non_uniform_dfl_minimal_onnx_export(tmp_path: Path):
    pytest.importorskip("onnx")
    module = NonUniformDFL(reg_max=16, gamma=1.5).eval()
    output_path = tmp_path / "non_uniform_dfl.onnx"
    torch.onnx.export(
        module,
        torch.randn(1, 64, 25),
        output_path,
        opset_version=12,
        input_names=["logits"],
        output_names=["distances"],
        dynamic_axes={
            "logits": {0: "batch", 2: "anchors"},
            "distances": {0: "batch", 2: "anchors"},
        },
    )
    assert output_path.stat().st_size > 0


def test_sudl_dfl_uniform_equivalence_sigma_and_empty_shape():
    torch.manual_seed(3)
    project = torch.arange(16, dtype=torch.float32)
    sudl = SUDLDFLoss(project, use_soft_label=False)
    original = DFLoss(reg_max=16)
    logits = torch.randn(5, 4, 16)
    targets = torch.rand(5, 4) * 14.5
    scale = torch.full((5, 1), 32.0)
    actual = sudl(logits, targets, scale)
    expected = original(logits.view(-1, 16), targets)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)
    assert (
        sudl._sigma(torch.tensor([[8.0]])).item()
        > sudl._sigma(torch.tensor([[64.0]])).item()
    )
    assert sudl(logits[:0], targets[:0], scale[:0]).shape == (0, 1)


def test_sudl_bbox_weight_bounds_normalization_empty_and_positive_backward():
    project = torch.arange(16, dtype=torch.float32)
    criterion = SUDLBboxLoss(reg_max=16, project=project)
    uncertainty = torch.tensor([[0.00], [0.05], [0.50]])
    scale = torch.tensor([[8.0], [24.0], [64.0]])
    base = torch.tensor([[0.2], [0.5], [1.0]])
    uncertainty_weight, _, extra = criterion._extra_weights(uncertainty, scale, base)
    assert uncertainty_weight.min() >= 1.0
    assert uncertainty_weight.max() <= 1.0 + criterion.uncertainty_gain
    assert torch.allclose((base * extra).sum(), base.sum(), atol=1e-6, rtol=1e-6)

    pred_dist = torch.randn(1, 4, 64, requires_grad=True)
    pred_bboxes = torch.tensor(
        [
            [
                [1.1, 1.0, 3.1, 3.0],
                [3.0, 3.1, 5.0, 5.1],
                [0.0, 0.0, 1.0, 1.0],
                [1.0, 1.0, 2.0, 2.0],
            ]
        ],
        requires_grad=True,
    )
    anchor_points = torch.tensor([[2.0, 2.0], [4.0, 4.0], [0.5, 0.5], [1.5, 1.5]])
    target_bboxes = torch.tensor(
        [
            [
                [1.0, 1.0, 3.0, 3.0],
                [3.0, 3.0, 5.0, 5.0],
                [0.0, 0.0, 1.0, 1.0],
                [1.0, 1.0, 2.0, 2.0],
            ]
        ]
    )
    target_scores = torch.zeros(1, 4, 4)
    target_scores[0, 0, 0] = 0.8
    target_scores[0, 1, 1] = 0.7
    fg_mask = torch.tensor([[True, True, False, False]])
    stride = torch.tensor([[8.0], [16.0], [32.0], [32.0]])
    loss_iou, loss_dfl = criterion(
        pred_dist,
        pred_bboxes,
        anchor_points,
        target_bboxes,
        target_scores,
        target_scores.sum(),
        fg_mask,
        stride_tensor=stride,
    )
    total = loss_iou + loss_dfl
    assert torch.isfinite(total)
    total.backward()
    assert pred_dist.grad is not None and torch.isfinite(pred_dist.grad).all()
    assert pred_bboxes.grad is not None and torch.isfinite(pred_bboxes.grad).all()

    empty_mask = torch.zeros_like(fg_mask)
    empty_iou, empty_dfl = criterion(
        pred_dist.detach().requires_grad_(True),
        pred_bboxes.detach().requires_grad_(True),
        anchor_points,
        target_bboxes,
        target_scores,
        torch.tensor(1.0),
        empty_mask,
        stride_tensor=stride,
    )
    assert empty_iou.item() == 0.0 and empty_dfl.item() == 0.0


def test_sudl_detect_replaces_only_dfl_projection():
    head = SUDLDetect(nc=4, dfl_gamma=1.5, ch=(16, 32, 64))
    keys = set(head.state_dict())
    assert "dfl.project" in keys
    assert "dfl.conv.weight" not in keys
    assert any(key.startswith("cv2.") for key in keys)
    assert any(key.startswith("cv3.") for key in keys)


def test_original_yolov13_loss_keeps_prototype_ciou_semantics():
    root = Path(__file__).resolve().parents[1]
    model = YOLO(str(root / "ultralytics/cfg/models/v13/yolov13n.yaml")).model
    criterion = v8DetectionLoss(model)
    assert type(criterion.bbox_loss) is BboxLoss
    assert isinstance(criterion.bbox_loss.dfl_loss, DFLoss)
    assert torch.equal(
        criterion.proj.cpu(), torch.arange(model.model[-1].reg_max, dtype=torch.float32)
    )
    assert isinstance(VarifocalLoss(), VarifocalLoss)
