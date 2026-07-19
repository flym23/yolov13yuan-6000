# Ultralytics 棣冩�?AGPL-3.0 License - https://ultralytics.com/license

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.metrics import OKS_SIGMA
from ultralytics.utils.ops import crop_mask, xywh2xyxy, xyxy2xywh
from ultralytics.utils.tal import RotatedTaskAlignedAssigner, TaskAlignedAssigner, dist2bbox, dist2rbox, make_anchors

from .metrics import bbox_iou, bbox_wiou_components, probiou
from .tal import bbox2dist


def _clear_ucra_aux_cache():
    """Clear UCRA auxiliary predictions produced during the current forward pass."""
    try:
        from ultralytics.nn.modules.block import _UCRABaseRefine

        cache = getattr(_UCRABaseRefine, "_ucra_aux_forward_cache", None)
        if cache is not None:
            cache.clear()
    except Exception:
        pass



class FocalLoss(nn.Module):
    """Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)."""

    def __init__(self):
        """Initializer for FocalLoss class with no parameters."""
        super().__init__()

    @staticmethod
    def forward(pred, label, gamma=1.5, alpha=0.25):
        """Calculates and updates confusion matrix for object detection/classification tasks."""
        loss = F.binary_cross_entropy_with_logits(pred, label, reduction="none")
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = pred.sigmoid()  # prob from logits
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
        modulating_factor = (1.0 - p_t) ** gamma
        loss *= modulating_factor
        if alpha > 0:
            alpha_factor = label * alpha + (1 - label) * (1 - alpha)
            loss *= alpha_factor
        return loss.mean(1).sum()


class DFLoss(nn.Module):
    """Criterion class for computing DFL losses during training."""

    def __init__(self, reg_max=16) -> None:
        """Initialize the DFL module."""
        super().__init__()
        self.reg_max = reg_max

    def __call__(self, pred_dist, target):
        """
        Return sum of left and right DFL losses.

        Distribution Focal Loss (DFL) proposed in Generalized Focal Loss
        https://ieeexplore.ieee.org/document/9792391
        """
        target = target.clamp_(0, self.reg_max - 1 - 0.01)
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (
            F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
            + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
        ).mean(-1, keepdim=True)


class SUDLDFLoss(nn.Module):
    """Distribution focal loss for non-uniform bins with optional scale-adaptive soft targets."""

    def __init__(
        self,
        project: torch.Tensor,
        use_soft_label: bool = True,
        sigma_base: float = 0.50,
        sigma_gain: float = 0.50,
        small_obj_px: float = 32.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        project = project.detach().float().clone()
        if project.ndim != 1 or project.numel() <= 1:
            raise ValueError("project must be a one-dimensional tensor with at least two values.")
        if not torch.all(project[1:] > project[:-1]):
            raise ValueError("project values must be strictly increasing.")
        if sigma_base <= 0:
            raise ValueError(f"sigma_base must be positive, got {sigma_base}.")
        if sigma_gain < 0:
            raise ValueError(f"sigma_gain must be non-negative, got {sigma_gain}.")
        if small_obj_px <= 0:
            raise ValueError(f"small_obj_px must be positive, got {small_obj_px}.")
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}.")

        self.register_buffer("project", project, persistent=True)
        self.use_soft_label = bool(use_soft_label)
        self.sigma_base = float(sigma_base)
        self.sigma_gain = float(sigma_gain)
        self.small_obj_px = float(small_obj_px)
        self.eps = float(eps)
        self.reg_max = int(project.numel())
        self.max_target = float(project[-1].item()) - 0.01

    def _sigma(self, target_scale_px: torch.Tensor) -> torch.Tensor:
        return self.sigma_base + self.sigma_gain * (
            1.0 - target_scale_px.float() / self.small_obj_px
        ).clamp(0.0, 1.0)

    def forward(
        self,
        pred_dist: torch.Tensor,
        target_ltrb: torch.Tensor,
        target_scale_px: torch.Tensor,
    ) -> torch.Tensor:
        if pred_dist.ndim != 3 or pred_dist.shape[1:] != (4, self.reg_max):
            raise ValueError(f"pred_dist must have shape [N, 4, {self.reg_max}], got {tuple(pred_dist.shape)}.")
        if target_ltrb.shape != pred_dist.shape[:2]:
            raise ValueError(f"target_ltrb must have shape [N, 4], got {tuple(target_ltrb.shape)}.")
        if target_scale_px.shape != (pred_dist.shape[0], 1):
            raise ValueError(f"target_scale_px must have shape [N, 1], got {tuple(target_scale_px.shape)}.")
        if pred_dist.shape[0] == 0:
            return pred_dist.new_zeros((0, 1))

        target = target_ltrb.float().clamp(0.0, self.max_target)
        log_prob = F.log_softmax(pred_dist.float(), dim=-1)
        project = self.project.to(device=target.device, dtype=torch.float32)

        if not self.use_soft_label:
            right = torch.searchsorted(project, target, right=False).clamp(1, project.numel() - 1)
            left = right - 1
            left_value = project[left]
            right_value = project[right]
            wr = (target - left_value) / (right_value - left_value).clamp_min(self.eps)
            wl = 1.0 - wr
            loss_left = -log_prob.gather(-1, left.unsqueeze(-1)).squeeze(-1)
            loss_right = -log_prob.gather(-1, right.unsqueeze(-1)).squeeze(-1)
            loss = loss_left * wl + loss_right * wr
            return loss.mean(dim=-1, keepdim=True).to(dtype=pred_dist.dtype)

        sigma = self._sigma(target_scale_px)
        distance = project.view(1, 1, -1) - target.unsqueeze(-1)
        target_prob = torch.exp(-0.5 * (distance / sigma.unsqueeze(-1).clamp_min(self.eps)).pow(2))
        target_prob = target_prob / target_prob.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        loss = -(target_prob * log_prob).sum(dim=-1)
        return loss.mean(dim=-1, keepdim=True).to(dtype=pred_dist.dtype)


class WiseIoULoss(nn.Module):
    """Stable WIoU loss with an EMA outlier-degree baseline."""

    def __init__(self, momentum=0.03, gamma=1.7, delta=4.0, eps=1e-7):
        super().__init__()
        self.momentum = momentum
        self.gamma = gamma
        self.delta = delta
        self.eps = eps
        self.register_buffer("iou_loss_mean", torch.tensor(1.0))

    def forward(self, box1, box2):
        iou, distance_term = bbox_wiou_components(box1, box2, xywh=False, eps=self.eps)
        iou_loss = (1.0 - iou).clamp(min=0.0)

        with torch.no_grad():
            if self.training and iou_loss.numel():
                mean = iou_loss.detach().mean()
                self.iou_loss_mean.mul_(1.0 - self.momentum).add_(mean * self.momentum)
            beta = (iou_loss.detach() / self.iou_loss_mean.clamp(min=self.eps)).clamp(min=0.0, max=10.0)
            gamma = torch.tensor(self.gamma, device=box1.device, dtype=box1.dtype)
            alpha = beta / (self.delta * torch.pow(gamma, beta - self.delta))
            alpha = alpha.clamp(min=0.05, max=1.50)

        return iou_loss + alpha * distance_term


def wise_iou_loss(box1, box2, eps=1e-7, gamma=1.7, delta=4.0):
    """Functional WIoU helper for standalone checks; training uses the stateful WiseIoULoss module."""
    iou, distance_term = bbox_wiou_components(box1, box2, xywh=False, eps=eps)
    iou_loss = (1.0 - iou).clamp(min=0.0)
    with torch.no_grad():
        beta = iou_loss / iou_loss.mean().clamp(min=eps)
        alpha = beta / (delta * torch.pow(torch.tensor(gamma, device=box1.device, dtype=box1.dtype), beta - delta))
        alpha = alpha.clamp(min=0.05, max=1.50)
    return iou_loss + alpha * distance_term


class BboxLoss(nn.Module):
    """Criterion class for computing training losses during training."""

    def __init__(self, reg_max=16, iou_type="wiou"):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None
        self.iou_type = iou_type
        if iou_type == "wiou":
            self.wiou_loss = WiseIoULoss()

    def forward(
        self,
        pred_dist,
        pred_bboxes,
        anchor_points,
        target_bboxes,
        target_scores,
        target_scores_sum,
        fg_mask,
        stride_tensor=None,
        imgsz=None,
    ):
        """IoU loss (WIoU or CIoU)."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        pred_pos = pred_bboxes[fg_mask]
        target_pos = target_bboxes[fg_mask]
        if self.iou_type == "wiou":
            loss_iou_val = self.wiou_loss(pred_pos, target_pos)
        else:
            # CIoU: bbox_iou returns CIoU value (higher is better); loss = 1 - CIoU
            iou = bbox_iou(pred_pos, target_pos, xywh=False, CIoU=True)
            loss_iou_val = 1.0 - iou
        loss_iou = (loss_iou_val * weight).sum() / target_scores_sum

        # DFL loss
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl


class SUDLBboxLoss(BboxLoss):
    """CIoU plus SUDL distribution loss with bounded uncertainty and small-object reweighting."""

    def __init__(
        self,
        reg_max,
        project,
        use_soft_label=True,
        sigma_base=0.50,
        sigma_gain=0.50,
        small_obj_px=32.0,
        use_uncertainty=True,
        uncertainty_gain=0.25,
        uncertainty_cap=0.20,
        use_scale_weight=True,
        scale_gain=0.25,
        eps=1e-6,
    ):
        super().__init__(reg_max=reg_max, iou_type="ciou")
        project = project.detach().float().clone()
        if project.ndim != 1 or project.numel() != int(reg_max):
            raise ValueError(f"project must contain exactly reg_max={reg_max} values.")
        if not torch.all(project[1:] > project[:-1]):
            raise ValueError("project values must be strictly increasing.")
        if uncertainty_gain < 0 or uncertainty_cap <= 0:
            raise ValueError("uncertainty_gain must be non-negative and uncertainty_cap must be positive.")
        if scale_gain < 0 or small_obj_px <= 0 or eps <= 0:
            raise ValueError("scale_gain must be non-negative; small_obj_px and eps must be positive.")

        self.reg_max = int(reg_max)
        self.use_uncertainty = bool(use_uncertainty)
        self.uncertainty_gain = float(uncertainty_gain)
        self.uncertainty_cap = float(uncertainty_cap)
        self.use_scale_weight = bool(use_scale_weight)
        self.scale_gain = float(scale_gain)
        self.small_obj_px = float(small_obj_px)
        self.eps = float(eps)
        self.register_buffer("project", project, persistent=True)
        self.dfl_loss = SUDLDFLoss(
            project=self.project,
            use_soft_label=use_soft_label,
            sigma_base=sigma_base,
            sigma_gain=sigma_gain,
            small_obj_px=small_obj_px,
            eps=eps,
        )

    def _extra_weights(self, box_uncertainty, target_scale_px, base_weight):
        """Return component and normalized DFL weights without changing their base-weighted total."""
        if self.use_uncertainty:
            uncertainty_weight = 1.0 + self.uncertainty_gain * (
                box_uncertainty.detach() / self.uncertainty_cap
            ).clamp(0.0, 1.0)
        else:
            uncertainty_weight = torch.ones_like(target_scale_px)
        if self.use_scale_weight:
            scale_weight = 1.0 + self.scale_gain * (
                1.0 - target_scale_px / self.small_obj_px
            ).clamp(0.0, 1.0)
        else:
            scale_weight = torch.ones_like(target_scale_px)

        extra = uncertainty_weight * scale_weight
        weighted_extra_mean = (base_weight * extra).sum() / base_weight.sum().clamp_min(self.eps)
        extra = extra / weighted_extra_mean.detach().clamp_min(self.eps)
        return uncertainty_weight, scale_weight, extra

    def forward(
        self,
        pred_dist,
        pred_bboxes,
        anchor_points,
        target_bboxes,
        target_scores,
        target_scores_sum,
        fg_mask,
        stride_tensor=None,
        imgsz=None,
    ):
        del imgsz
        if not fg_mask.any():
            return pred_bboxes.sum() * 0.0, pred_dist.sum() * 0.0
        if stride_tensor is None:
            raise ValueError("stride_tensor is required when SUDLBboxLoss has positive samples.")

        base_weight = target_scores.sum(-1)[fg_mask].reshape(-1, 1)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou).reshape(-1, 1) * base_weight).sum() / target_scores_sum

        target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max - 1)
        stride_map = stride_tensor.reshape(1, -1, 1).expand(fg_mask.shape[0], -1, -1)
        stride_pos = stride_map[fg_mask].to(device=target_bboxes.device, dtype=target_bboxes.dtype).reshape(-1, 1)
        target_pos_px = target_bboxes[fg_mask] * stride_pos
        target_wh_px = (target_pos_px[..., 2:] - target_pos_px[..., :2]).clamp_min(self.eps)
        target_scale_px = target_wh_px.prod(dim=-1, keepdim=True).sqrt()

        pred_pos = pred_dist[fg_mask].view(-1, 4, self.reg_max)
        probability = pred_pos.float().softmax(dim=-1)
        project = self.project.float().view(1, 1, self.reg_max)
        mean = (probability * project).sum(dim=-1)
        variance = (probability * (project - mean.unsqueeze(-1)).pow(2)).sum(dim=-1)
        box_uncertainty = variance.mean(dim=-1, keepdim=True) / float((self.reg_max - 1) ** 2)

        _, _, extra = self._extra_weights(box_uncertainty, target_scale_px, base_weight)
        dfl_weight = base_weight * extra

        dfl_item = self.dfl_loss(pred_pos, target_ltrb[fg_mask], target_scale_px)
        loss_dfl = (dfl_item * dfl_weight).sum() / target_scores_sum
        return loss_iou, loss_dfl


class RotatedBboxLoss(BboxLoss):
    """Criterion class for computing training losses during training."""

    def __init__(self, reg_max):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__(reg_max)

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """IoU loss."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = probiou(pred_bboxes[fg_mask], target_bboxes[fg_mask])
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, xywh2xyxy(target_bboxes[..., :4]), self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl


class KeypointLoss(nn.Module):
    """Criterion class for computing training losses."""

    def __init__(self, sigmas) -> None:
        """Initialize the KeypointLoss class."""
        super().__init__()
        self.sigmas = sigmas

    def forward(self, pred_kpts, gt_kpts, kpt_mask, area):
        """Calculates keypoint loss factor and Euclidean distance loss for predicted and actual keypoints."""
        d = (pred_kpts[..., 0] - gt_kpts[..., 0]).pow(2) + (pred_kpts[..., 1] - gt_kpts[..., 1]).pow(2)
        kpt_loss_factor = kpt_mask.shape[1] / (torch.sum(kpt_mask != 0, dim=1) + 1e-9)
        # e = d / (2 * (area * self.sigmas) ** 2 + 1e-9)  # from formula
        e = d / ((2 * self.sigmas).pow(2) * (area + 1e-9) * 2)  # from cocoeval
        return (kpt_loss_factor.view(-1, 1) * ((1 - torch.exp(-e)) * kpt_mask)).mean()


def _cfg_get(cfg, key, default=None):
    return cfg.get(key, default) if isinstance(cfg, dict) else getattr(cfg, key, default)


class v8DetectionLoss:
    def __init__(self, model, tal_topk=None):
        device = next(model.parameters()).device
        h = model.args

        m = model.model[-1]
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.hyp = h
        self.stride = m.stride
        self.nc = m.nc
        self.no = m.nc + m.reg_max * 4
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1
        self.assigner = TaskAlignedAssigner(topk=tal_topk or 10, num_classes=self.nc, alpha=0.5, beta=6.0)

        yaml_cfg = model.yaml if hasattr(model, "yaml") and isinstance(model.yaml, dict) else {}
        self.sudl_enabled = bool(getattr(m, "sudl_enabled", False) and yaml_cfg.get("sudl_enabled", True))
        self.nwd_gain = float(yaml_cfg.get("nwd_gain", _cfg_get(h, "nwd_gain", 0.0)))
        self.sudl_quality_calibration = bool(yaml_cfg.get("sudl_quality_calibration", True))
        self.sudl_quality_eta = float(yaml_cfg.get("sudl_quality_eta", 1.0))
        self.sudl_quality_floor = float(yaml_cfg.get("sudl_quality_floor", 0.75))
        if self.sudl_quality_eta < 0:
            raise ValueError(f"sudl_quality_eta must be non-negative, got {self.sudl_quality_eta}.")
        if not 0.0 < self.sudl_quality_floor <= 1.0:
            raise ValueError(f"sudl_quality_floor must be in (0, 1], got {self.sudl_quality_floor}.")

        use_wiou = True
        if yaml_cfg:
            use_wiou = yaml_cfg.get("wiou", _cfg_get(h, "wiou", True))
        elif not _cfg_get(h, "wiou", True):
            use_wiou = False
        iou_type = "wiou" if use_wiou else "ciou"
        if self.sudl_enabled:
            if self.nwd_gain > 0:
                raise ValueError("SUDL and NWD cannot be enabled at the same time.")
            self.proj = m.dfl.project.detach().to(device=device, dtype=torch.float32)
            self.bbox_loss = SUDLBboxLoss(
                reg_max=m.reg_max,
                project=self.proj,
                use_soft_label=bool(yaml_cfg.get("sudl_soft_label", True)),
                sigma_base=float(yaml_cfg.get("sudl_sigma_base", 0.50)),
                sigma_gain=float(yaml_cfg.get("sudl_sigma_gain", 0.50)),
                small_obj_px=float(yaml_cfg.get("sudl_small_obj_px", 32.0)),
                use_uncertainty=bool(yaml_cfg.get("sudl_uncertainty", True)),
                uncertainty_gain=float(yaml_cfg.get("sudl_uncertainty_gain", 0.25)),
                uncertainty_cap=float(yaml_cfg.get("sudl_uncertainty_cap", 0.20)),
                use_scale_weight=bool(yaml_cfg.get("sudl_scale_weight", True)),
                scale_gain=float(yaml_cfg.get("sudl_scale_gain", 0.25)),
            ).to(device)
        else:
            self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)
            self.bbox_loss = BboxLoss(m.reg_max, iou_type=iou_type).to(device)

        self.ucra_aux_gain = 0.0
        if yaml_cfg:
            self.ucra_aux_gain = float(yaml_cfg.get("ucra_aux", _cfg_get(h, "ucra_aux", 0.0)))
        else:
            self.ucra_aux_gain = float(_cfg_get(h, "ucra_aux", 0.0))

    def preprocess(self, targets, batch_size, scale_tensor):
        nl, ne = targets.shape
        if nl == 0:
            out = torch.zeros(batch_size, 0, ne - 1, device=self.device)
        else:
            i = targets[:, 0]
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)
            for j in range(batch_size):
                matches = i == j
                if n := matches.sum():
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        if self.use_dfl:
            b, a, c = pred_dist.shape
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def ucra_aux_loss(self, batch, batch_size):
        if self.ucra_aux_gain <= 0:
            _clear_ucra_aux_cache()
            return torch.zeros((), device=self.device)
        from ultralytics.nn.modules.block import _UCRABaseRefine

        cache = getattr(_UCRABaseRefine, "_ucra_aux_forward_cache", None)
        if not cache:
            return torch.zeros((), device=self.device)
        preds = [item["pred"] for item in cache if item["pred"].shape[0] == batch_size]
        _clear_ucra_aux_cache()
        if not preds:
            return torch.zeros((), device=self.device)

        batch_idx = batch["batch_idx"].view(-1).long().to(self.device)
        bboxes = batch["bboxes"].to(self.device)
        cls = batch["cls"].view(-1).long().to(self.device)
        level_losses = []
        for pred in preds:
            pred = pred.float()
            _, _, h, w = pred.shape
            obj_mask = torch.zeros((batch_size, 1, h, w), device=self.device, dtype=pred.dtype)
            weight_map = torch.ones_like(obj_mask)
            if bboxes.numel():
                for bi in range(batch_size):
                    inds = (batch_idx == bi).nonzero(as_tuple=False).flatten()
                    if inds.numel() == 0:
                        continue
                    for idx in inds:
                        x, y, bw, bh = bboxes[idx]
                        cx = x.clamp(0, 1) * (w - 1)
                        cy = y.clamp(0, 1) * (h - 1)
                        sx = (bw.clamp(min=1.0 / max(w, 1)) * w / 2.0).clamp(min=1.0)
                        sy = (bh.clamp(min=1.0 / max(h, 1)) * h / 2.0).clamp(min=1.0)
                        x0 = int(torch.clamp((cx - 3.0 * sx).floor(), 0, w - 1).item())
                        x1 = int(torch.clamp((cx + 3.0 * sx).ceil(), 0, w - 1).item())
                        y0 = int(torch.clamp((cy - 3.0 * sy).floor(), 0, h - 1).item())
                        y1 = int(torch.clamp((cy + 3.0 * sy).ceil(), 0, h - 1).item())
                        yy = torch.arange(y0, y1 + 1, device=self.device, dtype=pred.dtype).view(-1, 1)
                        xx = torch.arange(x0, x1 + 1, device=self.device, dtype=pred.dtype).view(1, -1)
                        gaussian = torch.exp(-0.5 * (((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2))
                        obj_mask[bi, 0, y0 : y1 + 1, x0 : x1 + 1] = torch.maximum(
                            obj_mask[bi, 0, y0 : y1 + 1, x0 : x1 + 1], gaussian
                        )
                        class_count = max((cls == cls[idx]).sum().item(), 1)
                        class_weight = (len(cls) / class_count) ** 0.5 if len(cls) else 1.0
                        weight_map[bi, 0, y0 : y1 + 1, x0 : x1 + 1] = torch.maximum(
                            weight_map[bi, 0, y0 : y1 + 1, x0 : x1 + 1],
                            torch.full_like(gaussian, min(class_weight, 3.0)),
                        )
            bce = F.binary_cross_entropy_with_logits(pred, obj_mask, reduction="none")
            bce_loss = (bce * weight_map).mean()
            pred_prob = pred.sigmoid()
            intersection = (pred_prob * obj_mask).sum(dim=(1, 2, 3))
            dice = 1.0 - (2.0 * intersection + 1.0) / (
                pred_prob.sum(dim=(1, 2, 3)) + obj_mask.sum(dim=(1, 2, 3)) + 1.0
            )
            level_losses.append(bce_loss + dice.mean())
        return torch.stack(level_losses).mean() * self.ucra_aux_gain

    def __call__(self, preds, batch):
        loss = torch.zeros(3, device=self.device)
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        if self.sudl_enabled and self.sudl_quality_calibration:
            with torch.no_grad():
                raw_probability = (
                    pred_distri.detach().view(batch_size, -1, 4, self.reg_max).float().softmax(dim=-1)
                )
                project = self.proj.float().view(1, 1, 1, self.reg_max)
                mean = (raw_probability * project).sum(dim=-1)
                variance = (raw_probability * (project - mean.unsqueeze(-1)).pow(2)).sum(dim=-1)
                uncertainty = variance.mean(dim=-1) / float((self.reg_max - 1) ** 2)
                quality = torch.exp(-self.sudl_quality_eta * uncertainty).clamp(
                    min=self.sudl_quality_floor,
                    max=1.0,
                )
            quality_factor = torch.where(
                fg_mask.unsqueeze(-1),
                quality.unsqueeze(-1).to(target_scores.dtype),
                torch.ones_like(target_scores[..., :1]),
            )
            target_scores = target_scores * quality_factor

        target_scores_sum = max(target_scores.sum(), 1)

        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum
        aux_loss = self.ucra_aux_loss(batch, batch_size)

        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes,
                target_scores,
                target_scores_sum,
                fg_mask,
                stride_tensor=stride_tensor,
                imgsz=imgsz,
            )

        loss[0] *= _cfg_get(self.hyp, "box", 7.5)
        loss[1] *= _cfg_get(self.hyp, "cls", 0.5)
        loss[1] += aux_loss
        loss[2] *= _cfg_get(self.hyp, "dfl", 1.5)

        return loss.sum() * batch_size, loss.detach()


class v8SegmentationLoss(v8DetectionLoss):
    def __init__(self, model):
        super().__init__(model)
        self.overlap = model.args.overlap_mask

    def __call__(self, preds, batch):
        loss = torch.zeros(4, device=self.device)
        feats, pred_masks, proto = preds if len(preds) == 3 else preds[1]
        batch_size, _, mask_h, mask_w = proto.shape
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_masks = pred_masks.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        try:
            batch_idx = batch["batch_idx"].view(-1, 1)
            targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 4), 2)
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        except RuntimeError as e:
            raise TypeError("ERROR segment dataset incorrectly formatted.") from e

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        loss[2] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum

        if fg_mask.sum():
            loss[0], loss[3] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes / stride_tensor,
                target_scores, target_scores_sum, fg_mask,
            )
            masks = batch["masks"].to(self.device).float()
            if tuple(masks.shape[-2:]) != (mask_h, mask_w):
                masks = F.interpolate(masks[None], (mask_h, mask_w), mode="nearest")[0]
            loss[1] = self.calculate_segmentation_loss(
                fg_mask, masks, target_gt_idx, target_bboxes, batch_idx, proto, pred_masks, imgsz, self.overlap
            )

        loss[0] *= _cfg_get(self.hyp, "box", 7.5)
        loss[1] *= _cfg_get(self.hyp, "box", 7.5)
        loss[2] *= _cfg_get(self.hyp, "cls", 0.5)
        loss[3] *= _cfg_get(self.hyp, "dfl", 1.5)

        return loss.sum() * batch_size, loss.detach()

    @staticmethod
    def calculate_segmentation_loss(
        fg_mask, masks, target_gt_idx, target_bboxes, batch_idx, proto, pred_masks, imgsz, overlap
    ):
        _, _, mask_h, mask_w = proto.shape
        loss = 0
        target_bboxes_normalized = target_bboxes / imgsz[[1, 0, 1, 0]]
        marea = xyxy2xywh(target_bboxes_normalized)[..., 2:].prod(2)
        mxyxy = target_bboxes_normalized * torch.tensor([mask_w, mask_h, mask_w, mask_h], device=proto.device)
        for bi in range(proto.shape[0]):
            if overlaps := (batch_idx == bi).nonzero(as_tuple=False):
                if 0 in overlaps.shape:
                    continue
            else:
                continue
            b_mask = masks[bi].unsqueeze(0)
            b_fg_mask = fg_mask[bi].unsqueeze(1)
            b_target_gt_idx = target_gt_idx[bi].unsqueeze(1)
            b_mxyxy = mxyxy[bi]
            b_marea = marea[bi]
            loss += single_mask_loss(
                b_mask, proto[bi], b_fg_mask, b_target_gt_idx, b_mxyxy, b_marea, pred_masks[bi], overlap=overlap
            )
        return loss / mask_h / mask_w / batch_idx.shape[0]

    @staticmethod
    def single_mask_loss(gt_mask, pred_proto, fg_mask, target_gt_idx, mxyxy, marea, pred_mask, overlap):
        if fg_mask.any():
            loss = F.binary_cross_entropy_with_logits(
                pred_mask[fg_mask], gt_mask[target_gt_idx[fg_mask]], reduction="none"
            )
            if overlap:
                loss = (loss.mean(-1) / marea[target_gt_idx[fg_mask]]).mean()
            else:
                loss = loss.mean()
            return loss
        return 0.0


class v8ClassificationLoss:
    def __call__(self, preds, batch):
        preds = preds[1] if isinstance(preds, (list, tuple)) else preds
        loss = F.cross_entropy(preds, batch["cls"], reduction="mean")
        loss_items = loss.detach()
        return loss, loss_items


class v8OBBLoss(v8DetectionLoss):
    def __init__(self, model):
        super().__init__(model)
        self.assigner = RotatedTaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = RotatedBboxLoss(self.reg_max).to(self.device)

    def preprocess(self, targets, batch_size, scale_tensor):
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 6, device=self.device)
        else:
            i = targets[:, 0]
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 6, device=self.device)
            for j in range(batch_size):
                matches = i == j
                if n := matches.sum():
                    bboxes = targets[matches, 2:]
                    bboxes[..., :4].mul_(scale_tensor)
                    out[j, :n] = torch.cat([targets[matches, 1:2], bboxes], dim=-1)
        return out

    def __call__(self, preds, batch):
        loss = torch.zeros(3, device=self.device)
        feats, pred_angle = preds if isinstance(preds[0], list) else preds[1]
        batch_size = pred_angle.shape[0]
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_angle = pred_angle.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        try:
            batch_idx = batch["batch_idx"].view(-1, 1)
            targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"].view(-1, 5)), 1)
            rw, rh = targets[:, 4] * imgsz[0].item(), targets[:, 5] * imgsz[1].item()
            targets = targets[(rw >= 2) & (rh >= 2)]
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 5), 2)
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        except RuntimeError as e:
            raise TypeError("ERROR OBB dataset incorrectly formatted.") from e

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri, pred_angle)

        bboxes_for_assigner = pred_bboxes.clone().detach()
        bboxes_for_assigner[..., :4] *= stride_tensor
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            bboxes_for_assigner.type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum

        if fg_mask.sum():
            target_bboxes[..., :4] /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )
        else:
            loss[0] += (pred_angle * 0).sum()

        loss[0] *= _cfg_get(self.hyp, "box", 7.5)
        loss[1] *= _cfg_get(self.hyp, "cls", 0.5)
        loss[2] *= _cfg_get(self.hyp, "dfl", 1.5)

        return loss.sum() * batch_size, loss.detach()

    def bbox_decode(self, anchor_points, pred_dist, pred_angle):
        if self.use_dfl:
            b, a, c = pred_dist.shape
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        return torch.cat((dist2rbox(pred_dist, pred_angle, anchor_points), pred_angle), dim=-1)


class v8PoseLoss(v8DetectionLoss):
    """Criterion class for computing training losses."""

    def __init__(self, model):
        """Initializes v8PoseLoss with model, assigner, and keypoint losses."""
        super().__init__(model)
        self.kpt_shape = model.model[-1].kpt_shape
        self.bce_pose = nn.BCEWithLogitsLoss()
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]
        sigmas = torch.from_numpy(OKS_SIGMA).to(self.device) if is_pose else torch.ones(nkpt, device=self.device) / nkpt
        self.keypoint_loss = KeypointLoss(sigmas=sigmas)

    def __call__(self, preds, batch):
        """Calculate the sum of the loss for box, cls, and kpts multiplied by batch size."""
        loss = torch.zeros(5, device=self.device)  # box, kpts, kpts_obj, cls, dfl
        feats, pred_kpts = preds if isinstance(preds[0], list) else preds[1]
        batch_size = feats[0].shape[0]

        pred_distri, pred_scores = torch.cat([xi.view(batch_size, self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_kpts = pred_kpts.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        batch_idx = batch["batch_idx"].view(-1, 1)
        targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        loss[3] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum

        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[4] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes,
                target_scores,
                target_scores_sum,
                fg_mask,
                stride_tensor=stride_tensor,
                imgsz=imgsz,
            )
            keypoints = batch["keypoints"].to(self.device).float().clone()
            keypoints[..., 0] *= imgsz[1]
            keypoints[..., 1] *= imgsz[0]

            loss[1], loss[2] = self.calculate_keypoints_loss(
                fg_mask, target_gt_idx, keypoints, batch_idx, stride_tensor, target_bboxes, pred_kpts
            )

        loss[0] *= _cfg_get(self.hyp, "box", 7.5)
        loss[1] *= _cfg_get(self.hyp, "pose", 12.0)
        loss[2] *= _cfg_get(self.hyp, "kobj", 1.0)
        loss[3] *= _cfg_get(self.hyp, "cls", 0.5)
        loss[4] *= _cfg_get(self.hyp, "dfl", 1.5)

        return loss.sum() * batch_size, loss.detach()

    @staticmethod
    def kpts_decode(anchor_points, pred_kpts):
        """Decodes predicted keypoints to image coordinates."""
        y = pred_kpts.clone()
        y[..., :2] *= 2.0
        y[..., 0] += anchor_points[:, [0]] - 0.5
        y[..., 1] += anchor_points[:, [1]] - 0.5
        return y

    def calculate_keypoints_loss(
        self, masks, target_gt_idx, keypoints, batch_idx, stride_tensor, target_bboxes, pred_kpts
    ):
        """Calculate the keypoints loss for the model."""
        batch_idx = batch_idx.flatten()
        batch_size = len(masks)

        max_kpts = torch.unique(batch_idx, return_counts=True)[1].max()

        batched_keypoints = torch.zeros(
            (batch_size, max_kpts, keypoints.shape[1], keypoints.shape[2]), device=keypoints.device
        )

        for i in range(batch_size):
            keypoints_i = keypoints[batch_idx == i]
            batched_keypoints[i, : keypoints_i.shape[0]] = keypoints_i

        target_gt_idx_expanded = target_gt_idx.unsqueeze(-1).unsqueeze(-1)

        selected_keypoints = batched_keypoints.gather(
            1, target_gt_idx_expanded.expand(-1, -1, keypoints.shape[1], keypoints.shape[2])
        )

        selected_keypoints /= stride_tensor.view(1, -1, 1, 1)

        kpts_loss = 0
        kpts_obj_loss = 0

        if masks.any():
            gt_kpt = selected_keypoints[masks]
            area = xyxy2xywh(target_bboxes[masks])[:, 2:].prod(1, keepdim=True)
            pred_kpt = pred_kpts[masks]
            kpt_mask = gt_kpt[..., 2] != 0 if gt_kpt.shape[-1] == 3 else torch.full_like(gt_kpt[..., 0], True)
            kpts_loss = self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)

            if pred_kpt.shape[-1] == 3:
                kpts_obj_loss = self.bce_pose(pred_kpt[..., 2], kpt_mask.float())

        return kpts_loss, kpts_obj_loss


class E2EDetectLoss:
    def __init__(self, model):
        self.one2many = v8DetectionLoss(model)
        self.one2one = v8DetectionLoss(model, tal_topk=1)

    def __call__(self, preds, batch):
        preds = preds[1] if isinstance(preds, tuple) else preds
        one2many = preds["one2many"]
        loss_one2many = self.one2many(one2many, batch)
        one2one = preds["one2one"]
        loss_one2one = self.one2one(one2one, batch)
        return loss_one2many[0] + loss_one2one[0], loss_one2many[1] + loss_one2one[1]


# single_mask_loss helper (must be module-level for v8SegmentationLoss to reference)
def single_mask_loss(gt_mask, pred_proto, fg_mask, target_gt_idx, mxyxy, marea, pred_mask, overlap):
    "Compute instance segmentation loss for a single image."
    if fg_mask.any():
        loss = F.binary_cross_entropy_with_logits(
            pred_mask[fg_mask], gt_mask[target_gt_idx[fg_mask]], reduction="none"
        )
        if overlap:
            loss = (loss.mean(-1) / marea[target_gt_idx[fg_mask]]).mean()
        else:
            loss = loss.mean()
        return loss
    return 0.0
