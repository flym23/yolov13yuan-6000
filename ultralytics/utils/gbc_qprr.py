from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ("GTBalancedCoveragePRRankLoss",)


class GTBalancedCoveragePRRankLoss(nn.Module):
    """Training-only GT-balanced ranking regularizer for dense object detection.

    Design goals:
      1. Preserve the successful QPRR positive-vs-negative ranking idea.
      2. Never sort positives belonging to different ground-truth instances.
      3. Give each GT approximately equal influence instead of pooling all same-class
         positives in an image, which can over-weight easy/crowded instances.
      4. Add a conservative per-GT coverage term: at least one well-localized
         positive should outrank dangerous safe negatives.

    The module never changes inference, Detect, NMS, TAL assignment, box loss, or DFL.
    All geometry/IoU quantities are detached.
    """

    def __init__(
        self,
        gain: float = 0.08,
        rank_margin: float = 0.50,
        sort_weight: float = 0.25,
        sort_margin: float = 1.00,
        coverage_weight: float = 0.35,
        coverage_margin: float = 0.50,
        coverage_topk: int = 2,
        coverage_neg_topk: int = 8,
        coverage_quality_floor: float = 0.50,
        coverage_temperature: float = 0.25,
        quality_floor: float = 0.35,
        quality_ceiling: float = 0.75,
        neg_topk: int = 32,
        max_pos_per_gt: int = 12,
        near_gt_expand: float = 1.25,
        small_thr: float = 0.050,
        small_temp: float = 0.0125,
        small_boost: float = 0.15,
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
        self.coverage_weight = float(coverage_weight)
        self.coverage_margin = float(coverage_margin)
        self.coverage_topk = int(coverage_topk)
        self.coverage_neg_topk = int(coverage_neg_topk)
        self.coverage_quality_floor = float(coverage_quality_floor)
        self.coverage_temperature = float(coverage_temperature)
        self.quality_floor = float(quality_floor)
        self.quality_ceiling = float(quality_ceiling)
        self.neg_topk = int(neg_topk)
        self.max_pos_per_gt = int(max_pos_per_gt)
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
        if min(self.rank_margin, self.sort_weight, self.sort_margin, self.coverage_weight, self.coverage_margin) < 0.0:
            raise ValueError("ranking/sorting/coverage weights and margins must be non-negative")
        if self.coverage_topk < 1 or self.coverage_neg_topk < 1:
            raise ValueError("coverage top-k values must be >= 1")
        if not 0.0 <= self.coverage_quality_floor <= 1.0:
            raise ValueError("coverage_quality_floor must be in [0, 1]")
        if self.coverage_temperature <= 0.0:
            raise ValueError("coverage_temperature must be positive")
        if not 0.0 <= self.quality_floor < self.quality_ceiling <= 1.0:
            raise ValueError("require 0 <= quality_floor < quality_ceiling <= 1")
        if self.neg_topk < 1 or self.max_pos_per_gt < 1:
            raise ValueError("neg_topk and max_pos_per_gt must be >= 1")
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

    def _small_weight(self, boxes_px: torch.Tensor, imgsz_hw: torch.Tensor) -> torch.Tensor:
        if boxes_px.ndim != 2 or boxes_px.shape[-1] != 4:
            raise ValueError("boxes_px must be [N, 4]")
        hw = imgsz_hw.detach().float().reshape(-1)
        if hw.numel() != 2:
            raise ValueError("imgsz_hw must contain [H, W]")
        h_img = hw[0].clamp_min(self.eps)
        w_img = hw[1].clamp_min(self.eps)
        boxes = boxes_px.detach().float()
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
        """Return background anchors outside an expanded region around every valid GT."""
        if anchor_points_px.ndim != 2 or anchor_points_px.shape[-1] != 2:
            raise ValueError("anchor_points_px must be [A, 2]")
        if gt_bboxes_px.ndim != 3 or gt_bboxes_px.shape[-1] != 4:
            raise ValueError("gt_bboxes_px must be [B, M, 4]")
        if fg_mask.shape != (gt_bboxes_px.shape[0], anchor_points_px.shape[0]):
            raise ValueError("fg_mask shape mismatch")

        valid_gt = mask_gt.squeeze(-1).bool() if mask_gt.ndim == 3 else mask_gt.bool()
        if valid_gt.shape != gt_bboxes_px.shape[:2]:
            raise ValueError("mask_gt shape mismatch")

        anchors = anchor_points_px.detach().float()
        near_gt = torch.zeros_like(fg_mask, dtype=torch.bool, device=fg_mask.device)
        for b in range(gt_bboxes_px.shape[0]):
            boxes = gt_bboxes_px[b][valid_gt[b]].detach().float()
            if boxes.numel() == 0:
                continue
            center = 0.5 * (boxes[:, :2] + boxes[:, 2:])
            half = 0.5 * (boxes[:, 2:] - boxes[:, :2]).clamp_min(0.0) * self.near_gt_expand
            lt, rb = center - half, center + half
            inside = ((anchors.unsqueeze(0) >= lt.unsqueeze(1)) & (anchors.unsqueeze(0) <= rb.unsqueeze(1))).all(-1)
            near_gt[b] = inside.any(dim=0)
        return (~fg_mask.bool()) & (~near_gt)

    def _normalized_smoothmax(self, values: torch.Tensor) -> torch.Tensor:
        """Temperature-smoothed max with no artificial dependence on set cardinality."""
        if values.numel() == 0:
            return values.sum() * 0.0
        v = values.float()
        tau = self.coverage_temperature
        return tau * (torch.logsumexp(v / tau, dim=0) - math.log(float(v.numel())))

    def _rank_one_gt(
        self,
        pos_logits: torch.Tensor,
        pos_quality: torch.Tensor,
        pos_boxes_px: torch.Tensor,
        neg_logits: torch.Tensor,
        imgsz_hw: torch.Tensor,
    ):
        """Compute rank/sort/coverage losses for exactly one GT instance."""
        zero = (pos_logits.sum() + neg_logits.sum()) * 0.0
        if pos_logits.numel() == 0 or neg_logits.numel() == 0:
            return zero, zero, zero, 0, 0, 0

        all_pos_logits = pos_logits.float()
        all_q = pos_quality.detach().float().clamp(0.0, 1.0)
        all_boxes = pos_boxes_px.detach().float()

        pos_prob = all_pos_logits.detach().sigmoid()
        quality_w = self._quality_gate(all_q)
        small_w = self._small_weight(all_boxes, imgsz_hw).to(all_q.device)
        pos_w = quality_w * (1.0 - pos_prob).clamp_min(0.0).pow(self.pos_gamma) * small_w

        active = pos_w > self.eps
        if not active.any():
            return zero, zero, zero, 0, 0, 0

        # Ranking/sorting positives: capped per GT, not per image/class.
        keep_logits = all_pos_logits
        keep_q = all_q
        keep_w = pos_w
        if keep_logits.numel() > self.max_pos_per_gt:
            keep = torch.topk(keep_w, k=self.max_pos_per_gt, largest=True).indices
            keep_logits = keep_logits[keep]
            keep_q = keep_q[keep]
            keep_w = keep_w[keep]

        # Same-class hard negatives. They are safe background anchors only.
        neg_f = neg_logits.float()
        kneg = min(self.neg_topk, neg_f.numel())
        neg_f = torch.topk(neg_f, k=kneg, largest=True).values
        neg_w = neg_f.detach().sigmoid().pow(self.neg_gamma).clamp_min(self.eps)

        diff = keep_logits[:, None] - neg_f[None, :]
        pair_loss = F.softplus(self.rank_margin - diff)
        pair_w = keep_w[:, None] * neg_w[None, :]
        denom = pair_w.sum()
        if float(denom.detach()) > self.eps:
            rank_loss = (pair_loss * pair_w).sum() / denom
            rank_pairs = int(((keep_w > self.eps).sum() * kneg).item())
        else:
            rank_loss = zero
            rank_pairs = 0

        # IMPORTANT: sorting is intra-GT only.
        qdiff = keep_q[:, None] - keep_q[None, :]
        sort_mask = qdiff > self.quality_gap
        if sort_mask.any():
            score_diff = keep_logits[:, None] - keep_logits[None, :]
            desired = self.sort_margin * qdiff
            sort_pair_loss = F.softplus(desired - score_diff)
            sw = torch.sqrt((keep_w[:, None] * keep_w[None, :]).clamp_min(0.0))
            sw = sw * qdiff.clamp_min(0.0) * sort_mask
            sden = sw.sum()
            if float(sden.detach()) > self.eps:
                sort_loss = (sort_pair_loss * sw).sum() / sden
                sort_pairs = int(sort_mask.sum().item())
            else:
                sort_loss, sort_pairs = zero, 0
        else:
            sort_loss, sort_pairs = zero, 0

        # Object-level coverage: one or two well-localized positives should outrank
        # the most dangerous safe negatives. Selection is by IoU, never by logit.
        cov_mask = all_q >= self.coverage_quality_floor
        if self.coverage_weight > 0.0 and cov_mask.any():
            cov_logits = all_pos_logits[cov_mask]
            cov_q = all_q[cov_mask]
            kc = min(self.coverage_topk, cov_logits.numel())
            idx = torch.topk(cov_q, k=kc, largest=True).indices
            z_pos = self._normalized_smoothmax(cov_logits[idx])

            kn = min(self.coverage_neg_topk, neg_f.numel())
            z_neg = self._normalized_smoothmax(neg_f[:kn])
            gt_small_weight = self._small_weight(all_boxes[:1], imgsz_hw).mean().to(z_pos.device)
            coverage_loss = F.softplus(self.coverage_margin - (z_pos - z_neg)) * gt_small_weight
            coverage_active = 1
        else:
            coverage_loss, coverage_active = zero, 0

        return rank_loss, sort_loss, coverage_loss, rank_pairs, sort_pairs, coverage_active

    def forward(
        self,
        pred_scores: torch.Tensor,
        target_scores: torch.Tensor,
        fg_mask: torch.Tensor,
        target_gt_idx: torch.Tensor,
        matched_iou: torch.Tensor,
        target_bboxes_px: torch.Tensor,
        anchor_points_px: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_bboxes_px: torch.Tensor,
        mask_gt: torch.Tensor,
        imgsz_hw: torch.Tensor,
        base_cls_loss: torch.Tensor,
    ):
        """Return calibrated auxiliary loss and diagnostics.

        Args:
            pred_scores: [B, A, C] raw classification logits.
            target_scores: [B, A, C] original TAL soft targets.
            fg_mask: [B, A] original TAL foreground mask.
            target_gt_idx: [B, A] local GT index returned by TAL.
            matched_iou: [N_fg] detached plain IoU, in fg_mask order.
            target_bboxes_px: [B, A, 4] assigned TAL boxes, pixel units.
            anchor_points_px: [A, 2] pixel anchor centers.
            gt_labels: [B, M, 1].
            gt_bboxes_px: [B, M, 4] padded GT boxes, pixel units.
            mask_gt: [B, M, 1] valid-GT mask.
            imgsz_hw: tensor [H, W].
            base_cls_loss: scalar original BCE classification loss before hyp.cls.
        """
        if pred_scores.ndim != 3:
            raise ValueError("pred_scores must be [B, A, C]")
        if target_scores.shape != pred_scores.shape:
            raise ValueError("target_scores shape mismatch")
        if fg_mask.shape != pred_scores.shape[:2] or target_gt_idx.shape != fg_mask.shape:
            raise ValueError("fg_mask/target_gt_idx shape mismatch")
        if target_bboxes_px.shape != (*pred_scores.shape[:2], 4):
            raise ValueError("target_bboxes_px shape mismatch")
        if gt_bboxes_px.ndim != 3 or gt_bboxes_px.shape[-1] != 4:
            raise ValueError("gt_bboxes_px must be [B, M, 4]")
        if gt_labels.shape[:2] != gt_bboxes_px.shape[:2]:
            raise ValueError("gt_labels shape mismatch")
        if not torch.is_tensor(base_cls_loss) or base_cls_loss.numel() != 1:
            raise ValueError("base_cls_loss must be scalar")

        zero = pred_scores.sum() * 0.0
        if not self.enabled:
            return zero, {
                "raw": zero.detach(),
                "calibration": zero.detach().new_tensor(1.0),
                "rank_pairs": 0,
                "sort_pairs": 0,
                "coverage_gts": 0,
                "valid_gts": 0,
                "active_gts": 0,
                "zero_positive_gts": 0,
                "same_class_multi_gt_gts": 0,
                "mean_pos_per_active_gt": zero.detach(),
                "safe_negative_fraction": zero.detach(),
                "auxiliary": zero.detach(),
            }

        valid_gt = mask_gt.squeeze(-1).bool() if mask_gt.ndim == 3 else mask_gt.bool()
        safe_neg = self.safe_negative_mask(anchor_points_px, gt_bboxes_px, mask_gt, fg_mask)

        quality_map = pred_scores.new_zeros(fg_mask.shape, dtype=torch.float32)
        miou = matched_iou.detach().float().reshape(-1)
        if miou.numel() != int(fg_mask.sum().item()):
            raise ValueError("matched_iou count must equal fg_mask.sum()")
        quality_map[fg_mask] = miou

        rank_terms, sort_terms, coverage_terms = [], [], []
        total_rank_pairs = total_sort_pairs = coverage_gts = 0
        valid_gts = active_gts = zero_positive_gts = same_class_multi_gt_gts = 0
        pos_counts = []

        for b in range(pred_scores.shape[0]):
            labels_b = gt_labels[b, :, 0].detach().long()
            valid_idx = torch.where(valid_gt[b])[0]
            if valid_idx.numel() == 0:
                continue

            valid_labels = labels_b[valid_idx]
            for c in valid_labels.unique():
                n_same = int((valid_labels == c).sum().item())
                if n_same > 1:
                    same_class_multi_gt_gts += n_same

            for g_t in valid_idx:
                g = int(g_t.item())
                valid_gts += 1
                c = int(labels_b[g].item())
                if c < 0 or c >= pred_scores.shape[2]:
                    raise ValueError(f"invalid GT class index {c}")

                pos_mask = fg_mask[b] & (target_gt_idx[b].long() == g)
                npos = int(pos_mask.sum().item())
                if npos == 0:
                    zero_positive_gts += 1
                    continue

                neg_mask = safe_neg[b]
                if not neg_mask.any():
                    continue

                rloss, sloss, closs, rp, sp, ca = self._rank_one_gt(
                    pred_scores[b, pos_mask, c],
                    quality_map[b, pos_mask],
                    target_bboxes_px[b, pos_mask],
                    pred_scores[b, neg_mask, c],
                    imgsz_hw,
                )
                if rp > 0:
                    rank_terms.append(rloss)
                    sort_terms.append(sloss)
                    active_gts += 1
                    pos_counts.append(float(npos))
                    total_rank_pairs += rp
                    total_sort_pairs += sp
                if ca:
                    coverage_terms.append(closs)
                    coverage_gts += 1

        if not rank_terms and not coverage_terms:
            return zero, {
                "raw": zero.detach(),
                "calibration": zero.detach().new_tensor(1.0),
                "rank_pairs": 0,
                "sort_pairs": 0,
                "coverage_gts": 0,
                "valid_gts": valid_gts,
                "active_gts": 0,
                "zero_positive_gts": zero_positive_gts,
                "same_class_multi_gt_gts": same_class_multi_gt_gts,
                "mean_pos_per_active_gt": zero.detach(),
                "safe_negative_fraction": safe_neg.float().mean().detach(),
                "auxiliary": zero.detach(),
            }

        rank_mean = torch.stack(rank_terms).mean() if rank_terms else zero
        sort_mean = torch.stack(sort_terms).mean() if sort_terms else zero
        coverage_mean = torch.stack(coverage_terms).mean() if coverage_terms else zero

        raw = rank_mean + self.sort_weight * sort_mean + self.coverage_weight * coverage_mean
        if float(raw.detach().abs()) <= self.eps:
            calibration = raw.detach().new_tensor(1.0)
            auxiliary = zero
        else:
            calibration = (
                base_cls_loss.detach().float() / raw.detach().float().clamp_min(self.eps)
            ).clamp(self.calibration_min, self.calibration_max)
            auxiliary = self.gain * raw * calibration.to(raw.dtype)

        diagnostics = {
            "raw": raw.detach(),
            "rank_mean": rank_mean.detach(),
            "sort_mean": sort_mean.detach(),
            "coverage_mean": coverage_mean.detach(),
            "calibration": calibration.detach(),
            "rank_pairs": total_rank_pairs,
            "sort_pairs": total_sort_pairs,
            "coverage_gts": coverage_gts,
            "valid_gts": valid_gts,
            "active_gts": active_gts,
            "zero_positive_gts": zero_positive_gts,
            "same_class_multi_gt_gts": same_class_multi_gt_gts,
            "mean_pos_per_active_gt": (
                raw.detach().new_tensor(sum(pos_counts) / len(pos_counts)) if pos_counts else raw.detach().new_tensor(0.0)
            ),
            "safe_negative_fraction": safe_neg.float().mean().detach(),
            "auxiliary": auxiliary.detach(),
        }
        return auxiliary, diagnostics
