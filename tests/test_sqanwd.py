import torch
import torch.nn as nn
from types import SimpleNamespace

from ultralytics.utils.loss import BboxLoss, SQANWDBboxLoss, v8DetectionLoss
from ultralytics.utils.sqanwd import ScaleQualityAdaptiveNWD


def test_round1_math_invariants():
    core = ScaleQualityAdaptiveNWD()

    target = torch.tensor(
        [
            [100.0, 100.0, 116.0, 116.0],
            [100.0, 100.0, 132.0, 132.0],
            [100.0, 100.0, 164.0, 164.0],
        ]
    )
    pred_same = target.clone()
    sim = core.nwd_similarity(pred_same, target, torch.tensor([640.0, 640.0]))
    assert torch.equal(sim, torch.ones_like(sim))

    pred_1 = target[:1].clone()
    pred_1[:, [0, 2]] += 1.0
    pred_4 = target[:1].clone()
    pred_4[:, [0, 2]] += 4.0
    s1 = core.nwd_similarity(pred_1, target[:1], torch.tensor([640.0, 640.0]))
    s4 = core.nwd_similarity(pred_4, target[:1], torch.tensor([640.0, 640.0]))
    assert float(s1) > float(s4)

    # Independent x/y resize invariance.
    scale = torch.tensor([2.0, 1.5, 2.0, 1.5])
    a = torch.tensor([[40.0, 50.0, 80.0, 90.0]])
    b = torch.tensor([[42.0, 48.0, 78.0, 94.0]])
    s_a = core.nwd_similarity(a, b, torch.tensor([480.0, 640.0]))
    s_b = core.nwd_similarity(
        a * scale,
        b * scale,
        torch.tensor([720.0, 1280.0]),
    )
    assert torch.allclose(s_a, s_b, atol=1e-7, rtol=1e-6)

    gate, ratio = core.size_gate(target, torch.tensor([640.0, 640.0]))
    assert float(gate[0]) > float(gate[1]) > float(gate[2])
    assert float(ratio[0]) < float(ratio[1]) < float(ratio[2])

    q = core.quality_gate(torch.tensor([[0.2], [0.55], [0.9]]))
    assert torch.allclose(q, torch.tensor([[1.0], [0.5], [0.0]]), atol=1e-6)

    off = ScaleQualityAdaptiveNWD(max_mix=0.0)
    ciou_loss = torch.tensor([[0.8], [0.3], [0.1]])
    plain_iou = 1.0 - ciou_loss
    weight = torch.tensor([[1.0], [2.0], [3.0]])
    hybrid, _ = off(
        ciou_loss,
        plain_iou,
        target,
        target,
        weight,
        torch.tensor([640.0, 640.0]),
    )
    assert torch.equal(hybrid, ciou_loss)

    print("ROUND1_PASS")


def test_round2_gradients_and_loss_scale():
    torch.manual_seed(7)
    core = ScaleQualityAdaptiveNWD()

    target = torch.tensor(
        [
            [100.0, 100.0, 116.0, 116.0],
            [180.0, 150.0, 220.0, 190.0],
            [260.0, 220.0, 340.0, 300.0],
            [400.0, 320.0, 520.0, 460.0],
        ],
        dtype=torch.float32,
    )
    pred = target.clone()
    pred[:, 0] += torch.tensor([4.0, 5.0, 8.0, 10.0])
    pred[:, 2] += torch.tensor([4.0, 5.0, 8.0, 10.0])
    pred.requires_grad_(True)

    # Synthetic CIoU/plain-IoU tensors for core-level verification.
    plain_iou = torch.tensor([[0.20], [0.45], [0.65], [0.85]])
    ciou_loss = torch.tensor([[0.85], [0.60], [0.40], [0.18]], requires_grad=True)
    weight = torch.tensor([[0.8], [1.0], [1.2], [1.0]])

    hybrid, diag = core(
        ciou_loss,
        plain_iou,
        pred,
        target,
        weight,
        torch.tensor([640.0, 640.0]),
    )

    assert torch.isfinite(hybrid).all()
    assert torch.isfinite(diag["nwd_similarity"]).all()
    assert float(diag["mix"].max()) <= 0.200001
    assert float(diag["mix"][-1]) == 0.0  # high-IoU branch is pure CIoU

    ciou_mean = core.weighted_mean(ciou_loss.detach(), weight)
    hybrid_mean = core.weighted_mean(hybrid.detach(), weight)
    ratio = float(hybrid_mean / ciou_mean)
    assert 0.79 <= ratio <= 1.26

    loss = (hybrid * weight).sum()
    loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
    assert ciou_loss.grad is not None
    assert torch.isfinite(ciou_loss.grad).all()

    # Low-IoU small box should receive more NWD mixing than a large/high-IoU box.
    assert float(diag["mix"][0]) > float(diag["mix"][1]) > float(diag["mix"][2])

    # Mixed-precision-facing path: core computes geometry in FP32 and casts back.
    pred_bf16 = pred.detach().to(torch.bfloat16)
    target_bf16 = target.to(torch.bfloat16)
    hybrid_bf16, diag_bf16 = core(
        ciou_loss.detach().to(torch.bfloat16),
        plain_iou.to(torch.bfloat16),
        pred_bf16,
        target_bf16,
        weight.to(torch.bfloat16),
        torch.tensor([640.0, 640.0], dtype=torch.bfloat16),
    )
    assert torch.isfinite(hybrid_bf16.float()).all()
    assert torch.isfinite(diag_bf16["nwd_similarity"].float()).all()

    print(
        "ROUND2_PASS",
        f"weighted_loss_ratio={ratio:.6f}",
        f"max_mix={float(diag['mix'].max()):.6f}",
    )


def test_detection_integration_and_exact_off_switch():
    class MinimalModel(nn.Module):
        def __init__(self, enabled):
            super().__init__()
            self.dummy = nn.Parameter(torch.zeros(()))
            self.model = [SimpleNamespace(stride=torch.tensor([8., 16., 32.]),
                                          nc=4, reg_max=16, no=68)]
            self.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)
            self.yaml = {'sqanwd': {'enabled': True, 'max_mix': 0.0}} if enabled else {}

    baseline = v8DetectionLoss(MinimalModel(False))
    candidate = v8DetectionLoss(MinimalModel(True))
    assert type(baseline.bbox_loss) is BboxLoss
    assert type(candidate.bbox_loss) is SQANWDBboxLoss

    pred_dist = torch.randn(1, 2, 64)
    pred_boxes = torch.tensor([[[1.0, 1.0, 3.0, 3.0], [2.0, 2.0, 4.0, 4.0]]])
    anchors = torch.tensor([[2.0, 2.0], [3.0, 3.0]])
    target_boxes = torch.tensor([[[1.1, 1.0, 3.1, 3.0], [2.0, 2.0, 4.0, 4.0]]])
    target_scores = torch.tensor([[[0.8, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]])
    foreground = torch.tensor([[True, False]])
    weight_sum = max(target_scores.sum(), 1)
    original = baseline.bbox_loss(pred_dist, pred_boxes, anchors, target_boxes,
                                  target_scores, weight_sum, foreground)
    disabled = candidate.bbox_loss(pred_dist, pred_boxes, anchors, target_boxes,
                                   target_scores, weight_sum, foreground,
                                   stride_tensor=torch.tensor([[8.0], [16.0]]),
                                   imgsz=torch.tensor([640.0, 640.0]))
    assert all(torch.equal(before, after) for before, after in zip(original, disabled))


if __name__ == "__main__":
    test_round1_math_invariants()
    test_round2_gradients_and_loss_scale()
    test_detection_integration_and_exact_off_switch()
