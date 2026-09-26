import torch

from ultralytics.utils.qprr import QualityGatedPRRankLoss


def _case(dtype=torch.float32):
    pred = torch.tensor(
        [[
            [0.0, -2.0], [1.0, -1.0], [-1.0, 0.3], [-2.0, 2.0],
            [0.8, -0.5], [-0.2, 0.5], [-3.0, -2.0], [2.2, -2.5],
        ]],
        dtype=dtype,
        requires_grad=True,
    )
    target = torch.zeros_like(pred)
    fg = torch.tensor([[1, 1, 0, 0, 1, 0, 0, 0]], dtype=torch.bool)
    target[0, 0, 0] = 0.6
    target[0, 1, 0] = 0.8
    target[0, 4, 0] = 0.4

    target_boxes = torch.zeros(1, 8, 4, dtype=dtype)
    target_boxes[0, 0] = torch.tensor([10, 10, 18, 18], dtype=dtype)
    target_boxes[0, 1] = torch.tensor([30, 30, 46, 46], dtype=dtype)
    target_boxes[0, 4] = torch.tensor([45, 10, 65, 30], dtype=dtype)

    anchors = torch.tensor(
        [[14, 14], [38, 38], [5, 5], [70, 70], [55, 20], [25, 60], [75, 10], [90, 90]],
        dtype=dtype,
    )
    gt = torch.tensor([[[10, 10, 18, 18], [30, 30, 46, 46], [45, 10, 65, 30]]], dtype=dtype)
    mask = torch.ones(1, 3, 1, dtype=torch.bool)
    iou = torch.tensor([[0.45], [0.85], [0.60]], dtype=dtype)
    imgsz = torch.tensor([100.0, 100.0], dtype=dtype)
    base_cls = torch.tensor(1.2, dtype=torch.float32)
    return pred, target, fg, iou, target_boxes, anchors, gt, mask, imgsz, base_cls


def test_round1_math_and_exact_off():
    qprr = QualityGatedPRRankLoss()
    pred, target, fg, iou, boxes, anchors, gt, mask, imgsz, base = _case()

    safe = qprr.safe_negative_mask(anchors, gt, mask, fg)
    assert safe.shape == fg.shape
    assert not safe[0, 0] and not safe[0, 1] and not safe[0, 4]

    off = QualityGatedPRRankLoss(gain=0.0)
    aux_off, diag_off = off(pred, target, fg, iou, boxes, anchors, gt, mask, imgsz, base)
    assert aux_off.detach().item() == 0.0
    assert diag_off["rank_pairs"] == 0

    small_weight = qprr._small_weight(
        torch.tensor([[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 40.0, 40.0]]),
        torch.tensor([100.0, 100.0]),
    )
    assert small_weight[0] > small_weight[1]

    # Quality floor is strict: poor boxes do not receive positive ranking weight.
    qgate = qprr._quality_gate(torch.tensor([0.20, 0.35, 0.55, 0.75, 0.90]))
    assert qgate[0] == 0 and qgate[1] == 0
    assert 0 < qgate[2] < 1
    assert qgate[3] == 1 and qgate[4] == 1

    aux0, _ = qprr(pred, target, fg, iou, boxes, anchors, gt, mask, imgsz, base)

    # Better ranking must reduce the auxiliary objective.
    better = pred.detach().clone()
    better[0, [0, 1, 4], 0] += 2.0
    better[0, [2, 3, 5, 6, 7], 0] -= 2.0
    better.requires_grad_(True)
    aux1, _ = qprr(better, target, fg, iou, boxes, anchors, gt, mask, imgsz, base)
    assert aux1 < aux0

    # Positive sorting: higher-IoU positive should prefer a larger class score.
    z_good = torch.tensor([0.0, 2.0, -0.5], requires_grad=True)
    z_bad = torch.tensor([2.0, 0.0, -0.5], requires_grad=True)
    q = torch.tensor([0.45, 0.85, 0.60])
    pos_boxes = torch.tensor([[0.0, 0.0, 8.0, 8.0], [0.0, 0.0, 16.0, 16.0], [0.0, 0.0, 20.0, 20.0]])
    neg = torch.tensor([0.5, 0.2, -0.1])
    _, sort_good, _, _ = qprr._rank_one_class(z_good, q, pos_boxes, neg, torch.tensor([100.0, 100.0]))
    _, sort_bad, _, _ = qprr._rank_one_class(z_bad, q, pos_boxes, neg, torch.tensor([100.0, 100.0]))
    assert sort_good < sort_bad


def test_round2_gradients_amp_and_empty_batch():
    qprr = QualityGatedPRRankLoss()
    pred, target, fg, iou, boxes, anchors, gt, mask, imgsz, base = _case()

    aux, diag = qprr(pred, target, fg, iou, boxes, anchors, gt, mask, imgsz, base)
    assert torch.isfinite(aux)
    assert qprr.calibration_min <= float(diag["calibration"]) <= qprr.calibration_max
    assert diag["rank_pairs"] > 0
    aux.backward()

    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()

    # Gradient descent must raise foreground target-class logits on average.
    pos_grad = pred.grad[0, [0, 1, 4], 0]
    assert pos_grad.mean() < 0

    # At least one hard safe negative must be pushed downward.
    safe = qprr.safe_negative_mask(anchors, gt, mask, fg)[0]
    neg_grad = pred.grad[0, torch.where(safe)[0], 0]
    assert neg_grad.max() > 0

    # Geometry/assignment inputs are detached by design.
    pred2, target, fg, iou, boxes, anchors, gt, mask, imgsz, base = _case()
    iou2 = iou.detach().clone().requires_grad_(True)
    boxes2 = boxes.detach().clone().requires_grad_(True)
    anchors2 = anchors.detach().clone().requires_grad_(True)
    gt2 = gt.detach().clone().requires_grad_(True)
    aux2, _ = qprr(pred2, target, fg, iou2, boxes2, anchors2, gt2, mask, imgsz, base)
    aux2.backward()
    assert iou2.grad is None
    assert boxes2.grad is None
    assert anchors2.grad is None
    assert gt2.grad is None

    # BF16-facing computation remains finite because pairwise math is FP32.
    pred3, target3, fg3, iou3, boxes3, anchors3, gt3, mask3, imgsz3, base3 = _case(torch.bfloat16)
    aux3, _ = qprr(pred3, target3, fg3, iou3, boxes3, anchors3, gt3, mask3, imgsz3, base3)
    assert torch.isfinite(aux3.float())
    aux3.backward()
    assert torch.isfinite(pred3.grad.float()).all()

    # No foreground: exact graph-safe zero.
    scores = torch.randn(2, 10, 3, requires_grad=True)
    t_scores = torch.zeros_like(scores)
    no_fg = torch.zeros(2, 10, dtype=torch.bool)
    t_boxes = torch.zeros(2, 10, 4)
    anc = torch.rand(10, 2) * 100
    gtb = torch.zeros(2, 0, 4)
    mgt = torch.zeros(2, 0, 1, dtype=torch.bool)
    zero, diag_zero = qprr(
        scores, t_scores, no_fg, torch.empty(0, 1), t_boxes, anc, gtb, mgt,
        torch.tensor([100.0, 100.0]), torch.tensor(1.0),
    )
    assert zero.detach().item() == 0.0
    assert diag_zero["rank_pairs"] == 0
    zero.backward()
    assert scores.grad is not None
