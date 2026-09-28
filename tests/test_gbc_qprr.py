import torch

from ultralytics.utils.gbc_qprr import GTBalancedCoveragePRRankLoss


def synthetic_case(dtype=torch.float32):
    # Two same-class GTs, each with two assigned positives; four safe negatives.
    pred = torch.tensor(
        [[
            [-0.4, -2.0], [0.2, -2.0],    # GT0 positives
            [-1.0, -2.0], [-0.2, -2.0],   # GT1 positives
            [0.4, -2.0], [0.1, -2.0], [-0.5, -2.0], [-1.5, -2.0],  # safe negatives
        ]],
        dtype=dtype,
        requires_grad=True,
    )
    target = torch.zeros_like(pred)
    fg = torch.tensor([[1,1,1,1,0,0,0,0]], dtype=torch.bool)
    target_gt_idx = torch.tensor([[0,0,1,1,0,0,0,0]], dtype=torch.long)
    target[0,0,0]=0.5; target[0,1,0]=0.8
    target[0,2,0]=0.5; target[0,3,0]=0.8

    assigned = torch.zeros(1,8,4,dtype=dtype)
    assigned[0,0]=assigned[0,1]=torch.tensor([10,10,22,22],dtype=dtype)
    assigned[0,2]=assigned[0,3]=torch.tensor([50,50,66,66],dtype=dtype)

    anchors=torch.tensor(
        [[14,14],[18,18],[54,54],[62,62],[5,80],[35,80],[80,15],[90,90]],
        dtype=dtype,
    )
    gt_labels=torch.tensor([[[0],[0]]],dtype=torch.float32)
    gt_boxes=torch.tensor([[[10,10,22,22],[50,50,66,66]]],dtype=dtype)
    mask_gt=torch.ones(1,2,1,dtype=torch.bool)
    # Within each GT, second positive is higher quality.
    iou=torch.tensor([[0.45],[0.82],[0.50],[0.88]],dtype=dtype)
    imgsz=torch.tensor([100.0,100.0],dtype=dtype)
    base=torch.tensor(1.2,dtype=torch.float32)
    return pred,target,fg,target_gt_idx,iou,assigned,anchors,gt_labels,gt_boxes,mask_gt,imgsz,base


def test_round1_grouping_exact_off_and_math():
    q = GTBalancedCoveragePRRankLoss()

    case = synthetic_case()
    pred,target,fg,tidx,iou,boxes,anchors,labels,gt,mask,imgsz,base = case

    # Same-class repeated instances must be recognized; sorting is still intra-GT.
    aux, diag = q(*case)
    assert torch.isfinite(aux)
    assert diag["same_class_multi_gt_gts"] == 2
    # Each GT has two positives => one directed quality-sort pair per GT.
    assert diag["sort_pairs"] == 2
    assert diag["active_gts"] == 2
    assert diag["coverage_gts"] == 2

    # Gain=0 is an exact graph-safe zero.
    off = GTBalancedCoveragePRRankLoss(gain=0.0)
    z, dz = off(*case)
    assert z.detach().item() == 0.0
    assert dz["rank_pairs"] == 0

    # No cross-GT sorting: swapping the absolute quality ordering BETWEEN GTs
    # must not create extra sort pairs.
    iou2 = torch.tensor([[0.70],[0.90],[0.40],[0.60]], dtype=iou.dtype)
    case2 = (pred,target,fg,tidx,iou2,boxes,anchors,labels,gt,mask,imgsz,base)
    _, d2 = q(*case2)
    assert d2["sort_pairs"] == 2

    # Better ranking for BOTH instances must reduce the objective.
    better = pred.detach().clone()
    better[0,[0,1,2,3],0] += 1.5
    better[0,[4,5,6,7],0] -= 1.5
    better.requires_grad_(True)
    better_case = (better,) + case[1:]
    aux_better, _ = q(*better_case)
    assert aux_better < aux

    # Small object receives greater bounded weight.
    sw = q._small_weight(
        torch.tensor([[0.,0.,8.,8.],[0.,0.,40.,40.]]),
        torch.tensor([100.,100.]),
    )
    assert sw[0] > sw[1]
    assert sw.max() <= 1.0 + q.small_boost + 1e-6

    # Quality floor/ceiling is strict.
    gate=q._quality_gate(torch.tensor([0.20,0.35,0.55,0.75,0.90]))
    assert gate[0] == 0 and gate[1] == 0
    assert 0 < gate[2] < 1
    assert gate[3] == 1 and gate[4] == 1


def test_round2_gradients_detach_bf16_and_empty():
    q = GTBalancedCoveragePRRankLoss()
    case = synthetic_case()
    pred,target,fg,tidx,iou,boxes,anchors,labels,gt,mask,imgsz,base = case
    aux, diag = q(*case)
    aux.backward()

    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
    # Target-class positives should on average be raised.
    assert pred.grad[0,[0,1,2,3],0].mean() < 0
    # At least one dangerous safe negative should be pushed down.
    safe=q.safe_negative_mask(anchors,gt,mask,fg)[0]
    assert pred.grad[0,torch.where(safe)[0],0].max() > 0
    assert q.calibration_min <= float(diag["calibration"]) <= q.calibration_max

    # Geometry and quality signals are detached.
    pred2,target2,fg2,tidx2,iou2,boxes2,anchors2,labels2,gt2,mask2,imgsz2,base2 = synthetic_case()
    iou2=iou2.detach().clone().requires_grad_(True)
    boxes2=boxes2.detach().clone().requires_grad_(True)
    anchors2=anchors2.detach().clone().requires_grad_(True)
    gt2=gt2.detach().clone().requires_grad_(True)
    aux2,_=q(pred2,target2,fg2,tidx2,iou2,boxes2,anchors2,labels2,gt2,mask2,imgsz2,base2)
    aux2.backward()
    assert iou2.grad is None
    assert boxes2.grad is None
    assert anchors2.grad is None
    assert gt2.grad is None

    # BF16-facing path remains finite.
    case3=synthetic_case(torch.bfloat16)
    aux3,_=q(*case3)
    assert torch.isfinite(aux3.float())
    aux3.backward()
    assert torch.isfinite(case3[0].grad.float()).all()

    # No foreground -> graph-safe zero.
    scores=torch.randn(2,10,3,requires_grad=True)
    t_scores=torch.zeros_like(scores)
    fg0=torch.zeros(2,10,dtype=torch.bool)
    tidx0=torch.zeros(2,10,dtype=torch.long)
    z,_=q(
        scores,t_scores,fg0,tidx0,torch.empty(0,1),
        torch.zeros(2,10,4),torch.rand(10,2)*100,
        torch.zeros(2,0,1),torch.zeros(2,0,4),
        torch.zeros(2,0,1,dtype=torch.bool),
        torch.tensor([100.,100.]),torch.tensor(1.0)
    )
    assert z.detach().item() == 0.0
    z.backward()
    assert scores.grad is not None
