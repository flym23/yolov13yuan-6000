from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ("QualityGatedPRRankLoss",)


class QualityGatedPRRankLoss(nn.Module):
    """Training-only pairwise ranking regularizer for dense object detection.

    The regularizer complements the detector's original BCE classification loss.
    It never changes inference code or model parameters.

    Two objectives are used:
      1) Rank trustworthy foreground target-class logits above hard, safe background
         logits of the same class.
      2) Sort foreground logits by detached localization quality.

    The regularizer is deliberately conservative:
      - positives are gated by detached matched IoU;
      - negatives near any GT box are ignored;
      - only top-k hard negatives and capped hard positives are used;
      - small-object positives receive a mild bounded weight boost;
      - magnitude is calibrated to the current BCE loss with detached statistics.
    """

    def __init__(
        self,
        gain: float = 0.08,
        rank_margin: float = 0.50,
        sort_weight: float = 0.25,
        sort_margin: float = 1.00,
        quality_floor: float = 0.35,
        quality_ceiling: float = 0.75,
        neg_topk: int = 32,
        max_pos: int = 64,
        near_gt_expand: float = 1.25,
        small_thr: float = 0.050,
        small_temp: float = 0.0125,
        small_boost: float = 0.20,
        pos_gamma: float = 1.0,
        neg_gamma: float = 1.0,
        quality_gap: float = 0.05,
        calibration_min: float = 0.25,
        calibration_max: float = 2.00,
        eps: float = 1e-9,
    ):
        super().__init__()
        self.gain = float(gain)
        self.rank_margin = float(rank_margin)
        self.sort_weight = float(sort_weight)
        self.sort_margin = float(sort_margin)
        self.quality_floor = float(quality_floor)
        self.quality_ceiling = float(quality_ceiling)
        self.neg_topk = int(neg_topk)
        self.max_pos = int(max_pos)
        self.near_gt_expand = float(near_gt_expand)
        self.small_thr = float(small_thr)
        self.small_temp = float(small_temp)
        self.small_boost = float(small_boost)
        self.pos_gamma = float(pos_gamma)
        self.neg_gamma = float(neg_gamma)
        self.quality_gap = float(quality_gap)
        self.calibration_min = float(calibration_min)
        self.calibration_max = float(calibration_max)
        self.eps = float(eps)

        if self.gain < 0.0:
            raise ValueError("gain must be non-negative")
        if self.rank_margin < 0.0 or self.sort_margin < 0.0 or self.sort_weight < 0.0:
            raise ValueError("ranking/sorting margins and weights must be non-negative")
        if not 0.0 <= self.quality_floor <= 1.0:
            raise ValueError("quality_floor must be in [0, 1]")
        if not self.quality_floor < self.quality_ceiling <= 1.0:
            raise ValueError("require quality_floor < quality_ceiling <= 1")
        if self.neg_topk < 1 or self.max_pos < 1:
            raise ValueError("neg_topk and max_pos must be >= 1")
        if self.near_gt_expand < 1.0:
            raise ValueError("near_gt_expand must be >= 1")
        if not 0.0 < self.small_thr < 1.0 or self.small_temp <= 0.0 or self.small_boost < 0.0:
            raise ValueError("invalid small-object weighting parameters")
        if self.pos_gamma < 0.0 or self.neg_gamma < 0.0 or self.quality_gap < 0.0:
            raise ValueError("gamma/gap parameters must be non-negative")
        if not 0.0 < self.calibration_min <= self.calibration_max:
            raise ValueError("invalid calibration clamp")
        if self.eps <= 0.0:
            raise ValueError("eps must be positive")

    @property
    def enabled(self) -> bool:
        return self.gain > 0.0

    def _quality_gate(self, matched_iou: torch.Tensor) -> torch.Tensor:
        q = matched_iou.detach().float().clamp(0.0, 1.0)
        return ((q - self.quality_floor) / (self.quality_ceiling - self.quality_floor)).clamp(0.0, 1.0)

    def _small_weight(self, assigned_boxes_px: torch.Tensor, imgsz_hw: torch.Tensor) -> torch.Tensor:
        if assigned_boxes_px.ndim != 2 or assigned_boxes_px.shape[-1] != 4:
            raise ValueError("assigned_boxes_px must be [N, 4]")
        hw = imgsz_hw.detach().float().reshape(-1)
        if hw.numel() != 2:
            raise ValueError("imgsz_hw must contain [H, W]")
        h_img = hw[0].clamp_min(self.eps)
        w_img = hw[1].clamp_min(self.eps)
        boxes = assigned_boxes_px.detach().float()
        bw = (boxes[:, 2] - boxes[:, 0]).clamp_min(0.0)
        bh = (boxes[:, 3] - boxes[:, 1]).clamp_min(0.0)
        ratio = torch.sqrt(((bw / w_img) * (bh / h_img)).clamp_min(0.0))
        gate = torch.sigmoid((self.small_thr - ratio) / self.small_temp)
        return 1.0 + self.small_boost * gate

    def safe_negative_mask(
        self,
        anchor_points_px: torch.Tensor,
        gt_bboxes_px: torch.Tensor,
        mask_gt: torch.Tensor,
        fg_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Background anchors outside an expanded region around every valid GT."""
        if anchor_points_px.ndim != 2 or anchor_points_px.shape[-1] != 2:
            raise ValueError("anchor_points_px must be [A, 2]")
        if gt_bboxes_px.ndim != 3 or gt_bboxes_px.shape[-1] != 4:
            raise ValueError("gt_bboxes_px must be [B, M, 4]")
        if fg_mask.ndim != 2 or fg_mask.shape[0] != gt_bboxes_px.shape[0]:
            raise ValueError("fg_mask must be [B, A]")
        if fg_mask.shape[1] != anchor_points_px.shape[0]:
            raise ValueError("anchor count mismatch")

        valid_gt = mask_gt.squeeze(-1).bool() if mask_gt.ndim == 3 else mask_gt.bool()
        if valid_gt.shape != gt_bboxes_px.shape[:2]:
            raise ValueError("mask_gt shape mismatch")

        device = anchor_points_px.device
        anchors = anchor_points_px.detach().float()
        near_gt = torch.zeros_like(fg_mask, dtype=torch.bool, device=device)

        for b in range(gt_bboxes_px.shape[0]):
            boxes = gt_bboxes_px[b][valid_gt[b]].detach().float()
            if boxes.numel() == 0:
                continue
            center = 0.5 * (boxes[:, :2] + boxes[:, 2:])
            half = 0.5 * (boxes[:, 2:] - boxes[:, :2]).clamp_min(0.0) * self.near_gt_expand
            lt = center - half
            rb = center + half
            # [M, A, 2]
            inside = ((anchors.unsqueeze(0) >= lt.unsqueeze(1)) & (anchors.unsqueeze(0) <= rb.unsqueeze(1))).all(-1)
            near_gt[b] = inside.any(dim=0)

        return (~fg_mask.bool()) & (~near_gt)

    def _rank_one_class(
        self,
        pos_logits: torch.Tensor,
        pos_quality: torch.Tensor,
        pos_boxes_px: torch.Tensor,
        neg_logits: torch.Tensor,
        imgsz_hw: torch.Tensor,
    ):
        """Return rank loss, sort loss and pair counts for one image/class."""
        if pos_logits.numel() == 0 or neg_logits.numel() == 0:
            zero = (pos_logits.sum() + neg_logits.sum()) * 0.0
            return zero, zero, 0, 0

        pos_logits_f = pos_logits.float()
        neg_logits_f = neg_logits.float()
        q = pos_quality.detach().float().clamp(0.0, 1.0)

        pos_prob = pos_logits_f.detach().sigmoid()
        quality_w = self._quality_gate(q)
        small_w = self._small_weight(pos_boxes_px, imgsz_hw).to(q.device)
        pos_w = quality_w * (1.0 - pos_prob).clamp_min(0.0).pow(self.pos_gamma) * small_w

        # Cap to the hardest trustworthy positives so runtime remains bounded.
        if pos_logits_f.numel() > self.max_pos:
            keep = torch.topk(pos_w, k=self.max_pos, largest=True).indices
            pos_logits_f = pos_logits_f[keep]
            q = q[keep]
            pos_w = pos_w[keep]

        # Hard safe negatives of the same class.
        k = min(self.neg_topk, neg_logits_f.numel())
        neg_logits_f = torch.topk(neg_logits_f, k=k, largest=True).values
        neg_w = neg_logits_f.detach().sigmoid().pow(self.neg_gamma).clamp_min(self.eps)

        diff = pos_logits_f[:, None] - neg_logits_f[None, :]
        rank_pair_loss = F.softplus(self.rank_margin - diff)
        rank_w = pos_w[:, None] * neg_w[None, :]
        rank_denom = rank_w.sum().clamp_min(self.eps)
        rank_loss = (rank_pair_loss * rank_w).sum() / rank_denom
        rank_pairs = int(rank_pair_loss.numel())

        # Sort higher-IoU positives above lower-IoU positives.
        qdiff = q[:, None] - q[None, :]
        sort_mask = qdiff > self.quality_gap
        if sort_mask.any():
            score_diff = pos_logits_f[:, None] - pos_logits_f[None, :]
            desired = self.sort_margin * qdiff
            sort_pair_loss = F.softplus(desired - score_diff)
            pair_w = torch.sqrt((pos_w[:, None] * pos_w[None, :]).clamp_min(0.0)) * qdiff.clamp_min(0.0)
            pair_w = pair_w * sort_mask
            sort_denom = pair_w.sum().clamp_min(self.eps)
            sort_loss = (sort_pair_loss * pair_w).sum() / sort_denom
            sort_pairs = int(sort_mask.sum().item())
        else:
            sort_loss = pos_logits_f.sum() * 0.0
            sort_pairs = 0

        return rank_loss, sort_loss, rank_pairs, sort_pairs

    def forward(
        self,
        pred_scores: torch.Tensor,
        target_scores: torch.Tensor,
        fg_mask: torch.Tensor,
        matched_iou: torch.Tensor,
        target_bboxes_px: torch.Tensor,
        anchor_points_px: torch.Tensor,
        gt_bboxes_px: torch.Tensor,
        mask_gt: torch.Tensor,
        imgsz_hw: torch.Tensor,
        base_cls_loss: torch.Tensor,
    ):
        """Return calibrated auxiliary loss and detached diagnostics.

        Args:
            pred_scores: [B, A, C] raw classification logits.
            target_scores: [B, A, C] TAL targets.
            fg_mask: [B, A] foreground mask.
            matched_iou: [N_fg, 1] or [N_fg] detached plain IoU in fg_mask order.
            target_bboxes_px: [B, A, 4] assigned target boxes in pixel units.
            anchor_points_px: [A, 2] anchor centers in pixel units.
            gt_bboxes_px: [B, M, 4] padded GT boxes in pixel units.
            mask_gt: [B, M, 1] valid-GT mask.
            imgsz_hw: [2] tensor [H, W].
            base_cls_loss: scalar original BCE classification loss before hyp.cls.
        """
        if pred_scores.ndim != 3:
            raise ValueError("pred_scores must be [B, A, C]")
        if target_scores.shape != pred_scores.shape:
            raise ValueError("target_scores shape mismatch")
        if fg_mask.shape != pred_scores.shape[:2]:
            raise ValueError("fg_mask shape mismatch")
        if target_bboxes_px.shape != (*pred_scores.shape[:2], 4):
            raise ValueError("target_bboxes_px shape mismatch")
        if not torch.is_tensor(base_cls_loss) or base_cls_loss.numel() != 1:
            raise ValueError("base_cls_loss must be a scalar tensor")

        if not self.enabled:
            zero = pred_scores.sum() * 0.0
            return zero, {
                "raw": zero.detach(),
                "calibration": zero.detach().new_tensor(1.0),
                "rank_pairs": 0,
                "sort_pairs": 0,
                "safe_negative_fraction": zero.detach().new_tensor(0.0),
            }

        safe_neg = self.safe_negative_mask(anchor_points_px, gt_bboxes_px, mask_gt, fg_mask)
        quality_map = pred_scores.new_zeros(fg_mask.shape, dtype=torch.float32)
        matched_iou_f = matched_iou.detach().float().reshape(-1)
        if matched_iou_f.numel() != int(fg_mask.sum().item()):
            raise ValueError("matched_iou count must equal fg_mask.sum()")
        quality_map[fg_mask] = matched_iou_f

        target_cls = target_scores.detach().argmax(dim=-1)
        rank_terms, sort_terms = [], []
        total_rank_pairs = 0
        total_sort_pairs = 0

        for b in range(pred_scores.shape[0]):
            for c in range(pred_scores.shape[2]):
                pos_mask = fg_mask[b] & (target_cls[b] == c)
                if not pos_mask.any():
                    continue
                neg_mask = safe_neg[b]
                if not neg_mask.any():
                    continue

                rank_loss, sort_loss, rp, sp = self._rank_one_class(
                    pred_scores[b, pos_mask, c],
                    quality_map[b, pos_mask],
                    target_bboxes_px[b, pos_mask],
                    pred_scores[b, neg_mask, c],
                    imgsz_hw,
                )
                rank_terms.append(rank_loss)
                sort_terms.append(sort_loss)
                total_rank_pairs += rp
                total_sort_pairs += sp

        if not rank_terms:
            zero = pred_scores.sum() * 0.0
            return zero, {
                "raw": zero.detach(),
                "calibration": zero.detach().new_tensor(1.0),
                "rank_pairs": 0,
                "sort_pairs": 0,
                "safe_negative_fraction": safe_neg.float().mean().detach(),
            }

        rank_mean = torch.stack(rank_terms).mean()
        sort_mean = torch.stack(sort_terms).mean() if sort_terms else rank_mean * 0.0
        raw = rank_mean + self.sort_weight * sort_mean

        calibration = (
            base_cls_loss.detach().float() / raw.detach().float().clamp_min(self.eps)
        ).clamp(self.calibration_min, self.calibration_max)
        calibrated = raw * calibration.to(raw.dtype)
        auxiliary = self.gain * calibrated

        diagnostics = {
            "raw": raw.detach(),
            "rank_mean": rank_mean.detach(),
            "sort_mean": sort_mean.detach(),
            "calibration": calibration.detach(),
            "rank_pairs": total_rank_pairs,
            "sort_pairs": total_sort_pairs,
            "safe_negative_fraction": safe_neg.float().mean().detach(),
            "auxiliary": auxiliary.detach(),
        }
        return auxiliary, diagnostics
