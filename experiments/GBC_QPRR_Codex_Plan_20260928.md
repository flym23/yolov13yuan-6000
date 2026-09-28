# GBC-QPRR-YOLOv13：基于 QPRR 全量 URPC2020 结果的下一步 Codex 执行方案

日期：2026-09-28

## 0. 结论先行

QPRR 是目前为止第一次在完整 URPC2020 上给出比较清楚的、可重复的 P/F1 正信号。

| 方案 | P | R | F1 | F1 std | mAP50-95 | AP_S |
|---|---:|---:|---:|---:|---:|---:|
| A0 | 82.975 | 77.125 | 79.939 | 0.069 | 50.168 | 15.370 |
| A4 | 83.529 | 76.700 | 79.967 | 0.187 | 50.022 | 16.287 |
| B1 | 82.443 | 77.374 | 79.826 | 0.260 | 50.201 | 15.652 |
| C1 | 82.746 | 77.104 | 79.823 | 0.250 | 50.148 | 14.804 |
| C2 | 82.736 | 76.840 | 79.678 | 0.227 | 50.034 | 15.062 |
| D1 | 83.408 | 76.997 | 80.074 | 0.125 | 50.089 | 15.726 |
| D2 | 82.972 | 77.283 | 80.024 | 0.026 | 50.171 | 15.603 |
| D3 | 83.365 | 77.088 | 80.101 | 0.255 | 50.163 | 15.587 |

### D3 = A0 + QPRR v1

相对 A0：
- P +0.3895 pp
- R -0.0372 pp
- F1 +0.1620 pp
- mAP50-95 -0.0048 pp
- AP_S +0.2169 pp

F1 paired-seed：
[0.3824, 0.1004, 0.0031] pp

**3/3 seed F1 都为正。**

因此 D3 已经可以作为论文中的 balanced-F1 有效改进：
它不增加推理参数/GFLOPs，mean P 和 mean F1 上升，mAP50-95 基本不变。

### D1 = A4 + QPRR v1

相对 A4：
- P -0.1206 pp
- R +0.2976 pp
- F1 +0.1077 pp

F1 paired-seed：
[0.1921, 0.1139, 0.0172] pp

同样 **3/3 F1 为正**；同时 Recall 较 A4 恢复约 +0.298 pp。

所以 QPRR 的主要机制已经被两种不同 base 重复支持。

### D2 = B1 + QPRR v1

相对 A0：
- P -0.0036 pp
- R +0.1583 pp
- F1 +0.0842 pp
- mAP50-95 +0.0035 pp

D2 的 F1 std 仅 0.0262 pp，是目前非常稳定的 balanced 方案，但增益量较小。

---

# 1. 当前应保留的论文结构

## 1.1 A4：Precision-oriented

A4 仍保留：
- P 相对 A0 +0.554 pp；
- AP_S 相对 A0 +0.917 pp。

## 1.2 D3：balanced-F1 oriented

D3 是当前最干净的第二条贡献：
- architecture 与 A0 完全相同；
- training-only QPRR；
- inference cost 0；
- F1 3/3 seed 正增益。

## 1.3 D1：Precision + F1 composite

D1 的 P 相对 A0仍有 +0.433 pp，
F1 相对 A0有 +0.135 pp，
且 AP_S 相对 A0有 +0.356 pp。

所以 D1 也可以作为组合实验保留。

---

# 2. 为什么还要继续改 QPRR

QPRR v1 已经有效，但它有一个代码级结构性问题：

当前实现按：

```text
image -> class
```

聚合所有 positives。

如果一张图里有多个**同类别 GT**，它们的 positives 会被放到同一个池中。

这带来两个问题：

1. easy GT 拥有更多/更强 positives 时，会在平均 pairwise loss 中占更大权重；
2. `sort loss` 会把不同实例的 positives 按 IoU 相互排序。

第二点尤其不合理：

```text
GT-A 的 IoU=0.85
GT-B 的 IoU=0.55
```

并不意味着 GT-A 的分类置信度就应该系统性高于 GT-B。

这会天然偏向 easy objects / high-quality objects，
而困难、遮挡、小目标实例更容易被压低。

这与现有实验非常吻合：
- QPRR 明显提高 P/F1；
- 但 D3 的 mean R 几乎没有提高；
- D1 虽恢复 A4 Recall，但仍未明显超过 A0 Recall。

---

# 3. 新方案：GBC-QPRR

名称：

**GBC-QPRR = Ground-Truth-Balanced Coverage Precision-Recall Ranking**

核心目标：

```text
保留 QPRR 已验证的 Precision/F1 优势
+
把优化单位从“image/class”改为“每一个 GT instance”
+
增加极保守的 object-level coverage 约束
+
重点补 Recall
```

推理仍然完全不变。

---

# 4. 三个核心修改

## 4.1 GT-balanced ranking

利用 TAL 已经返回但当前 detection loss 丢弃的：

```python
target_gt_idx
```

把 positives 按真正的 GT identity 分组：

```text
image
  ├─ GT0
  ├─ GT1
  └─ GT2
```

每个 GT 独立计算 ranking loss，最后按 GT 平均。

因此一个有 8 个 positives 的 easy object，
不会比只有 2 个 positives 的困难 object 获得 4 倍话语权。

## 4.2 Intra-GT sorting

sorting 只允许发生在：

```text
同一个 GT 的 positives 内部
```

禁止不同 GT 间用 IoU 比较 classification score。

这是相对 QPRR-v1 最重要的逻辑修复。

## 4.3 GT coverage

对每个 GT：

1. 只取 IoU >= 0.50 的定位可靠 positives；
2. 按 IoU 选最多 2 个，不按 classification score 选；
3. 和同类别最危险的 safe hard negatives 比较；
4. 要求至少一个高质量 positive 在 ranking 上能压过 hard negatives。

形式：

```text
L_cov = softplus(
    margin - (smoothmax(z_pos_good) - smoothmax(z_neg_hard))
)
```

这不是把所有正样本一起抬高，
而是确保每个 GT 至少有一个可靠 detection candidate。

其目标直接对应 Recall。

---

# 5. 为什么不修改 TAL

虽然样本分配本身非常重要，但本轮不直接改变 TAL：

- 现有 D3 已证明原 TAL + QPRR 可以工作；
- 修改 topk/alpha/beta 会同时影响 cls/box/DFL，归因变差；
- 当前首先修复的是 QPRR 自身的 instance pooling 问题。

因此：

```text
TAL assignment 不变
box positives 不变
DFL positives 不变
target_scores 不变
```

GBC-QPRR 仍只是 training-only classification regularizer。

---

# 6. 完整模块

新建：

`ultralytics/utils/gbc_qprr.py`

```python
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

```

---

# 7. loss.py 精确集成

```text
# Integration patch for ultralytics/utils/loss.py
#
# Existing QPRR v1 MUST remain untouched for reproducibility of D1/D2/D3.
# Add GBC-QPRR as a separate optional training-only path.

# 1) Import near existing QPRR import:
from ultralytics.utils.gbc_qprr import GTBalancedCoveragePRRankLoss

# 2) In v8DetectionLoss.__init__, AFTER existing QPRR setup:
gbc_cfg = getattr(model, "yaml", {}).get("gbc_qprr", {}) or {}
self.gbc_qprr_cfg = dict(gbc_cfg)
self.use_gbc_qprr = bool(gbc_cfg.get("enabled", False)) and float(gbc_cfg.get("gain", 0.0)) > 0.0

if self.use_qprr and self.use_gbc_qprr:
    raise ValueError("qprr and gbc_qprr are mutually exclusive; enable only one.")

self.gbc_qprr = (
    GTBalancedCoveragePRRankLoss(
        gain=float(gbc_cfg.get("gain", 0.08)),
        rank_margin=float(gbc_cfg.get("rank_margin", 0.50)),
        sort_weight=float(gbc_cfg.get("sort_weight", 0.25)),
        sort_margin=float(gbc_cfg.get("sort_margin", 1.00)),
        coverage_weight=float(gbc_cfg.get("coverage_weight", 0.35)),
        coverage_margin=float(gbc_cfg.get("coverage_margin", 0.50)),
        coverage_topk=int(gbc_cfg.get("coverage_topk", 2)),
        coverage_neg_topk=int(gbc_cfg.get("coverage_neg_topk", 8)),
        coverage_quality_floor=float(gbc_cfg.get("coverage_quality_floor", 0.50)),
        coverage_temperature=float(gbc_cfg.get("coverage_temperature", 0.25)),
        quality_floor=float(gbc_cfg.get("quality_floor", 0.35)),
        quality_ceiling=float(gbc_cfg.get("quality_ceiling", 0.75)),
        neg_topk=int(gbc_cfg.get("neg_topk", 32)),
        max_pos_per_gt=int(gbc_cfg.get("max_pos_per_gt", 12)),
        near_gt_expand=float(gbc_cfg.get("near_gt_expand", 1.25)),
        small_thr=float(gbc_cfg.get("small_thr", 0.050)),
        small_temp=float(gbc_cfg.get("small_temp", 0.0125)),
        small_boost=float(gbc_cfg.get("small_boost", 0.15)),
        pos_gamma=float(gbc_cfg.get("pos_gamma", 1.0)),
        neg_gamma=float(gbc_cfg.get("neg_gamma", 1.0)),
        quality_gap=float(gbc_cfg.get("quality_gap", 0.05)),
        calibration_min=float(gbc_cfg.get("calibration_min", 0.25)),
        calibration_max=float(gbc_cfg.get("calibration_max", 2.00)),
        eps=float(gbc_cfg.get("eps", 1e-9)),
    ).to(device)
    if self.use_gbc_qprr
    else None
)
self.last_gbc_qprr_diagnostics = {}

# 3) Capture target_gt_idx from the EXISTING TAL call.
# Change only the final placeholder:
_, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
    pred_scores.detach().sigmoid(),
    (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
    anchor_points * stride_tensor,
    gt_labels,
    gt_bboxes,
    mask_gt,
)

# 4) KEEP original BCE:
base_cls_loss = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum
loss[1] = base_cls_loss

# 5) Inside `if fg_mask.sum():`, keep the pixel-space clone BEFORE
#    `target_bboxes /= stride_tensor`.
#
# Replace the existing QPRR-only block with a mutually-exclusive block:
if self.use_qprr or self.use_gbc_qprr:
    target_bboxes_px = target_bboxes.detach().clone()
    pred_bboxes_px = pred_bboxes * stride_tensor
    matched_iou = bbox_iou(
        pred_bboxes_px[fg_mask],
        target_bboxes_px[fg_mask],
        xywh=False,
        CIoU=False,
    ).detach().clamp(0.0, 1.0).reshape(-1, 1)

    if self.use_gbc_qprr:
        aux, diag = self.gbc_qprr(
            pred_scores=pred_scores,
            target_scores=target_scores,
            fg_mask=fg_mask,
            target_gt_idx=target_gt_idx,
            matched_iou=matched_iou,
            target_bboxes_px=target_bboxes_px,
            anchor_points_px=(anchor_points * stride_tensor).detach(),
            gt_labels=gt_labels.detach(),
            gt_bboxes_px=gt_bboxes.detach(),
            mask_gt=mask_gt.detach(),
            imgsz_hw=imgsz.detach(),
            base_cls_loss=base_cls_loss,
        )
        loss[1] = loss[1] + aux
        self.last_gbc_qprr_diagnostics = diag
        self.last_qprr_diagnostics = {}
    else:
        qprr_aux, qprr_diag = self.qprr(
            pred_scores=pred_scores,
            target_scores=target_scores,
            fg_mask=fg_mask,
            matched_iou=matched_iou,
            target_bboxes_px=target_bboxes_px,
            anchor_points_px=(anchor_points * stride_tensor).detach(),
            gt_bboxes_px=gt_bboxes.detach(),
            mask_gt=mask_gt.detach(),
            imgsz_hw=imgsz.detach(),
            base_cls_loss=base_cls_loss,
        )
        loss[1] = loss[1] + qprr_aux
        self.last_qprr_diagnostics = qprr_diag
        self.last_gbc_qprr_diagnostics = {}
else:
    self.last_qprr_diagnostics = {}
    self.last_gbc_qprr_diagnostics = {}

# 6) If fg_mask is empty, clear both diagnostics:
else:
    self.last_qprr_diagnostics = {}
    self.last_gbc_qprr_diagnostics = {}

# 7) Absolutely do NOT change:
# - TaskAlignedAssigner internals/topk/alpha/beta
# - target_scores
# - fg_mask used by box/DFL
# - SQANWD code
# - Detect / NMS / validation
# - optimizer or augmentation recipe

```

特别强调：

- `target_gt_idx` 直接使用 TAL 返回值；
- 不能自行重算 GT assignment；
- pixel-space `target_bboxes` 仍必须 `.clone()` 后再参与 auxiliary loss；
- qprr v1 与 gbc_qprr 必须互斥；
- D1/D2/D3 的旧代码不能删，必须保证可复现。

---

# 8. 实验设计

## E0：exact-off

A0 + GBC-QPRR gain=0。

只做代码等价性：

```text
A0 state == E0 state
A0 loss == E0 loss
A0 shared gradients == E0 gradients
```

## E1：instance-balanced only

A0 + GBC-QPRR：

```text
coverage_weight = 0
```

回答：

> 仅修复跨 GT pooling/sorting 是否改善 D3？

## E2：full GBC-QPRR

A0 +：

```text
GT-balanced rank
intra-GT sort
coverage
```

这是主候选。

## E3：A4 + full GBC-QPRR

只有 E2 三 seed 通过后执行。

目标：

```text
A4 Precision 优势
+
GBC-QPRR Recall/F1
```

---

# 9. YAML

## E0

```yaml
nc: 4

gbc_qprr:
  enabled: true
  gain: 0.00
  rank_margin: 0.50
  sort_weight: 0.25
  sort_margin: 1.00
  coverage_weight: 0.00
  coverage_margin: 0.50
  coverage_topk: 2
  coverage_neg_topk: 8
  coverage_quality_floor: 0.50
  coverage_temperature: 0.25
  quality_floor: 0.35
  quality_ceiling: 0.75
  neg_topk: 32
  max_pos_per_gt: 12
  near_gt_expand: 1.25
  small_thr: 0.050
  small_temp: 0.0125
  small_boost: 0.15
  pos_gamma: 1.0
  neg_gamma: 1.0
  quality_gap: 0.05
  calibration_min: 0.25
  calibration_max: 2.00
  eps: 1.0e-9

scales:
  n: [0.50, 0.25, 1024]
  s: [0.50, 0.50, 1024]
  l: [1.00, 1.00, 512]
  x: [1.00, 1.50, 512]

backbone:
  - [-1, 1, Conv,  [64, 3, 2]]
  - [-1, 1, Conv,  [128, 3, 2, 1, 2]]
  - [-1, 2, DSC3k2,  [256, False, 0.25]]
  - [-1, 1, Conv,  [256, 3, 2, 1, 4]]
  - [-1, 2, DSC3k2,  [512, False, 0.25]]
  - [-1, 1, DSConv,  [512, 3, 2]]
  - [-1, 4, A2C2f, [512, True, 4]]
  - [-1, 1, DSConv,  [1024, 3, 2]]
  - [-1, 4, A2C2f, [1024, True, 1]]

head:
  - [[4, 6, 8], 2, HyperACE, [512, 8, True, True, 0.5, 1, "both"]]
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [9, 1, DownsampleConv, []]
  - [[6, 9], 1, FullPAD_Tunnel, []]
  - [[4, 10], 1, FullPAD_Tunnel, []]
  - [[8, 11], 1, FullPAD_Tunnel, []]

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 12], 1, Concat, [1]]
  - [-1, 2, DSC3k2, [512, True]]
  - [[-1, 9], 1, FullPAD_Tunnel, []]

  - [17, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 13], 1, Concat, [1]]
  - [-1, 2, DSC3k2, [256, True]]
  - [10, 1, Conv, [256, 1, 1]]
  - [[21, 22], 1, FullPAD_Tunnel, []]

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 18], 1, Concat, [1]]
  - [-1, 2, DSC3k2, [512, True]]
  - [[-1, 9], 1, FullPAD_Tunnel, []]

  - [26, 1, Conv, [512, 3, 2]]
  - [[-1, 14], 1, Concat, [1]]
  - [-1, 2, DSC3k2, [1024, True]]
  - [[-1, 11], 1, FullPAD_Tunnel, []]

  - [[23, 27, 31], 1, Detect, [nc]]

```

## E1

```yaml
nc: 4

gbc_qprr:
  enabled: true
  gain: 0.08
  rank_margin: 0.50
  sort_weight: 0.25
  sort_margin: 1.00
  coverage_weight: 0.00
  coverage_margin: 0.50
  coverage_topk: 2
  coverage_neg_topk: 8
  coverage_quality_floor: 0.50
  coverage_temperature: 0.25
  quality_floor: 0.35
  quality_ceiling: 0.75
  neg_topk: 32
  max_pos_per_gt: 12
  near_gt_expand: 1.25
  small_thr: 0.050
  small_temp: 0.0125
  small_boost: 0.15
  pos_gamma: 1.0
  neg_gamma: 1.0
  quality_gap: 0.05
  calibration_min: 0.25
  calibration_max: 2.00
  eps: 1.0e-9

scales:
  n: [0.50, 0.25, 1024]
  s: [0.50, 0.50, 1024]
  l: [1.00, 1.00, 512]
  x: [1.00, 1.50, 512]

backbone:
  - [-1, 1, Conv,  [64, 3, 2]]
  - [-1, 1, Conv,  [128, 3, 2, 1, 2]]
  - [-1, 2, DSC3k2,  [256, False, 0.25]]
  - [-1, 1, Conv,  [256, 3, 2, 1, 4]]
  - [-1, 2, DSC3k2,  [512, False, 0.25]]
  - [-1, 1, DSConv,  [512, 3, 2]]
  - [-1, 4, A2C2f, [512, True, 4]]
  - [-1, 1, DSConv,  [1024, 3, 2]]
  - [-1, 4, A2C2f, [1024, True, 1]]

head:
  - [[4, 6, 8], 2, HyperACE, [512, 8, True, True, 0.5, 1, "both"]]
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [9, 1, DownsampleConv, []]
  - [[6, 9], 1, FullPAD_Tunnel, []]
  - [[4, 10], 1, FullPAD_Tunnel, []]
  - [[8, 11], 1, FullPAD_Tunnel, []]

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 12], 1, Concat, [1]]
  - [-1, 2, DSC3k2, [512, True]]
  - [[-1, 9], 1, FullPAD_Tunnel, []]

  - [17, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 13], 1, Concat, [1]]
  - [-1, 2, DSC3k2, [256, True]]
  - [10, 1, Conv, [256, 1, 1]]
  - [[21, 22], 1, FullPAD_Tunnel, []]

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 18], 1, Concat, [1]]
  - [-1, 2, DSC3k2, [512, True]]
  - [[-1, 9], 1, FullPAD_Tunnel, []]

  - [26, 1, Conv, [512, 3, 2]]
  - [[-1, 14], 1, Concat, [1]]
  - [-1, 2, DSC3k2, [1024, True]]
  - [[-1, 11], 1, FullPAD_Tunnel, []]

  - [[23, 27, 31], 1, Detect, [nc]]

```

## E2

```yaml
nc: 4

gbc_qprr:
  enabled: true
  gain: 0.08
  rank_margin: 0.50
  sort_weight: 0.25
  sort_margin: 1.00
  coverage_weight: 0.35
  coverage_margin: 0.50
  coverage_topk: 2
  coverage_neg_topk: 8
  coverage_quality_floor: 0.50
  coverage_temperature: 0.25
  quality_floor: 0.35
  quality_ceiling: 0.75
  neg_topk: 32
  max_pos_per_gt: 12
  near_gt_expand: 1.25
  small_thr: 0.050
  small_temp: 0.0125
  small_boost: 0.15
  pos_gamma: 1.0
  neg_gamma: 1.0
  quality_gap: 0.05
  calibration_min: 0.25
  calibration_max: 2.00
  eps: 1.0e-9

scales:
  n: [0.50, 0.25, 1024]
  s: [0.50, 0.50, 1024]
  l: [1.00, 1.00, 512]
  x: [1.00, 1.50, 512]

backbone:
  - [-1, 1, Conv,  [64, 3, 2]]
  - [-1, 1, Conv,  [128, 3, 2, 1, 2]]
  - [-1, 2, DSC3k2,  [256, False, 0.25]]
  - [-1, 1, Conv,  [256, 3, 2, 1, 4]]
  - [-1, 2, DSC3k2,  [512, False, 0.25]]
  - [-1, 1, DSConv,  [512, 3, 2]]
  - [-1, 4, A2C2f, [512, True, 4]]
  - [-1, 1, DSConv,  [1024, 3, 2]]
  - [-1, 4, A2C2f, [1024, True, 1]]

head:
  - [[4, 6, 8], 2, HyperACE, [512, 8, True, True, 0.5, 1, "both"]]
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [9, 1, DownsampleConv, []]
  - [[6, 9], 1, FullPAD_Tunnel, []]
  - [[4, 10], 1, FullPAD_Tunnel, []]
  - [[8, 11], 1, FullPAD_Tunnel, []]

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 12], 1, Concat, [1]]
  - [-1, 2, DSC3k2, [512, True]]
  - [[-1, 9], 1, FullPAD_Tunnel, []]

  - [17, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 13], 1, Concat, [1]]
  - [-1, 2, DSC3k2, [256, True]]
  - [10, 1, Conv, [256, 1, 1]]
  - [[21, 22], 1, FullPAD_Tunnel, []]

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 18], 1, Concat, [1]]
  - [-1, 2, DSC3k2, [512, True]]
  - [[-1, 9], 1, FullPAD_Tunnel, []]

  - [26, 1, Conv, [512, 3, 2]]
  - [[-1, 14], 1, Concat, [1]]
  - [-1, 2, DSC3k2, [1024, True]]
  - [[-1, 11], 1, FullPAD_Tunnel, []]

  - [[23, 27, 31], 1, Detect, [nc]]

```

## E3

```yaml
nc: 4

gbc_qprr:
  enabled: true
  gain: 0.08
  rank_margin: 0.50
  sort_weight: 0.25
  sort_margin: 1.00
  coverage_weight: 0.35
  coverage_margin: 0.50
  coverage_topk: 2
  coverage_neg_topk: 8
  coverage_quality_floor: 0.50
  coverage_temperature: 0.25
  quality_floor: 0.35
  quality_ceiling: 0.75
  neg_topk: 32
  max_pos_per_gt: 12
  near_gt_expand: 1.25
  small_thr: 0.050
  small_temp: 0.0125
  small_boost: 0.15
  pos_gamma: 1.0
  neg_gamma: 1.0
  quality_gap: 0.05
  calibration_min: 0.25
  calibration_max: 2.00
  eps: 1.0e-9

scales:
  n: [0.50, 0.25, 1024]
  s: [0.50, 0.50, 1024]
  l: [1.00, 1.00, 512]
  x: [1.00, 1.50, 512]

backbone:
  - [-1, 1, Conv,  [64, 3, 2]]
  - [-1, 1, Conv,  [128, 3, 2, 1, 2]]
  - [-1, 2, DSC3k2,  [256, False, 0.25]]
  - [-1, 1, Conv,  [256, 3, 2, 1, 4]]
  - [-1, 2, DSC3k2,  [512, False, 0.25]]
  - [-1, 1, DSConv,  [512, 3, 2]]
  - [-1, 4, A2C2f, [512, True, 4]]
  - [-1, 1, DSConv,  [1024, 3, 2]]
  - [-1, 4, A2C2f, [1024, True, 1]]

head:
  - [[4, 6, 8], 2, HyperACE, [512, 8, True, True, 0.5, 1, "both"]]
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [9, 1, DownsampleConv, []]
  - [[6, 9], 1, FullPAD_Tunnel, []]
  - [[4, 10], 1, FullPAD_Tunnel, []]
  - [[8, 11], 1, FullPAD_Tunnel, []]

  - [[14, 12], 1, UCRA1v2, []]
  - [[-1, 12], 1, Concat, [1]]
  - [-1, 2, DSC3k2, [512, True]]
  - [[-1, 9], 1, FullPAD_Tunnel, []]

  - [[17, 13], 1, UCRA2v2, []]
  - [[-1, 13], 1, Concat, [1]]
  - [-1, 2, DSC3k2, [256, True]]
  - [10, 1, Conv, [256, 1, 1]]
  - [[21, 22], 1, FullPAD_Tunnel, []]

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 18], 1, Concat, [1]]
  - [-1, 2, DSC3k2, [512, True]]
  - [[-1, 9], 1, FullPAD_Tunnel, []]

  - [26, 1, Conv, [512, 3, 2]]
  - [[-1, 14], 1, Concat, [1]]
  - [-1, 2, DSC3k2, [1024, True]]
  - [[-1, 11], 1, FullPAD_Tunnel, []]

  - [[23, 27, 31], 1, Detect, [nc]]

```

---

# 10. 为什么不再只跑 seed0 做筛选

这次 QPRR 已经给出一个非常重要的教训。

D3 vs A0 的 Recall paired delta 是：

```text
[1.257, -1.0533, -0.3154] pp
```

seed0 看起来是非常大的 Recall 增益，
但 seed1/seed2 方向完全不同。

所以以后：

```text
禁止只凭 seed0 决定一个 P/R/F1 模块是否成功
```

E1/E2 首轮至少同时跑：

```text
seed 0 + seed 1
```

再决定是否补 seed2。

---

# 11. 两轮代码自检

## Round 1

新建：

`tests/test_gbc_qprr.py`

```python
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

```

本地独立模块实际检查：

- gain=0 exact zero；
- same-class multi-GT 识别；
- cross-GT sorting 被严格禁止；
- intra-GT sorting pair 数正确；
- improved ranking -> loss下降；
- small-object weight bounded；
- IoU quality gate 正确。

## Round 2

同一测试继续检查：

- positive target-class gradient 方向；
- hard-negative gradient 方向；
- IoU/box/anchor/GT 全 detach；
- calibration bounded；
- BF16-facing finite；
- no-FG graph-safe zero。

---

# 12. 仓库级自检

新建：

`tests/verify_gbc_qprr_repo.py`

```python
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
A0 = ROOT / "ultralytics/cfg/models/v13/yolov13.yaml"
E0 = ROOT / "ultralytics/cfg/models/v13/yolov13n-gbc-qprr-e0-off.yaml"
E1 = ROOT / "ultralytics/cfg/models/v13/yolov13n-gbc-qprr-e1-instance.yaml"
E2 = ROOT / "ultralytics/cfg/models/v13/yolov13n-gbc-qprr-e2-full.yaml"
E3 = ROOT / "ultralytics/cfg/models/v13/yolov13n-a4-gbc-qprr-e3.yaml"


def clone_batch(batch):
    return {k:(v.clone() if torch.is_tensor(v) else v) for k,v in batch.items()}


def state_equal(a,b):
    sa,sb=a.state_dict(),b.state_dict()
    assert sa.keys()==sb.keys()
    for k in sa:
        assert torch.equal(sa[k],sb[k]),k


def main():
    seed=20260928
    torch.manual_seed(seed)
    a0=YOLO(str(A0)).model
    torch.manual_seed(seed)
    e0=YOLO(str(E0)).model
    state_equal(a0,e0)

    assert len(a0.model)==len(e0.model)==33
    assert e0.model[-1].__class__.__name__=="Detect"
    assert e0.model[-1].f==[23,27,31]
    assert torch.equal(e0.model[-1].stride.cpu(),torch.tensor([8.,16.,32.]))

    batch={
        "img":torch.randn(2,3,256,256),
        "batch_idx":torch.tensor([0.,0.,0.,1.,1.]),
        "cls":torch.tensor([[0.],[0.],[1.],[2.],[3.]]),
        "bboxes":torch.tensor([
            [0.25,0.25,0.08,0.08],
            [0.55,0.25,0.07,0.07],  # repeated same-class GT in image0
            [0.70,0.65,0.14,0.12],
            [0.35,0.70,0.08,0.07],
            [0.75,0.25,0.14,0.12],
        ]),
    }

    a0.train(); e0.train()
    a0.zero_grad(set_to_none=True); e0.zero_grad(set_to_none=True)
    la,ia=a0.loss(clone_batch(batch))
    le,ie=e0.loss(clone_batch(batch))
    assert torch.equal(la,le)
    assert torch.equal(ia,ie)
    la.backward(); le.backward()
    pa=dict(a0.named_parameters()); pe=dict(e0.named_parameters())
    for k in pa:
        ga,ge=pa[k].grad,pe[k].grad
        assert (ga is None)==(ge is None),k
        if ga is not None:
            assert torch.equal(ga,ge),k

    # E1/E2: training-only loss path, no model parameters or inference graph added.
    e1=YOLO(str(E1)).model
    e2=YOLO(str(E2)).model
    assert sum(p.numel() for p in e1.parameters())==sum(p.numel() for p in a0.parameters())
    assert sum(p.numel() for p in e2.parameters())==sum(p.numel() for p in a0.parameters())

    for model in (e1,e2):
        model.train()
        model.zero_grad(set_to_none=True)
        loss,items=model.loss(clone_batch(batch))
        assert torch.isfinite(loss)
        assert torch.isfinite(items).all()
        loss.backward()
        assert getattr(model.criterion,"use_gbc_qprr",False)
        diag=model.criterion.last_gbc_qprr_diagnostics
        assert diag
        assert diag["valid_gts"] >= 5
        assert diag["rank_pairs"] > 0
        assert torch.isfinite(diag["auxiliary"])

    # E2 must actually activate coverage on this batch.
    assert e2.criterion.last_gbc_qprr_diagnostics["coverage_gts"] > 0

    # E3 uses A4 neck but still original Detect.
    e3=YOLO(str(E3)).model
    assert len(e3.model)==33
    assert e3.model[15].__class__.__name__=="UCRA1v2"
    assert e3.model[19].__class__.__name__=="UCRA2v2"
    assert e3.model[-1].__class__.__name__=="Detect"

    # Inference-facing graph is unchanged by the training-only loss.
    for model in (e1,e2,e3):
        model.eval()
        with torch.no_grad():
            out=model(torch.randn(1,3,640,640))
        assert out is not None

    print("GBC-QPRR repository verification passed.")


if __name__=="__main__":
    main()

```

必须运行：

```bash
python -m py_compile ultralytics/utils/gbc_qprr.py
pytest -q tests/test_gbc_qprr.py
python tests/verify_gbc_qprr_repo.py
```

---

# 13. 接受标准

当前下一步不再只对 A0。

## E2 相对 D3

理想 balanced-F1 模式：

1. mean F1 > D3；
2. 至少 2/3 seed F1 > D3；
3. mean R > D3；
4. mean P 不低于 A0；
5. AP_S 相对 D3下降 <=0.20 pp；
6. mAP50-95 相对 D3下降 <=0.20 pp。

## 若 balanced 不成立，也允许论文有效模式

Precision 模式：
- P >= A0 +0.30 pp；
- 2/3 seed 为正；
- F1 相对 D3下降 <=0.10 pp。

Recall 模式：
- R >= A0 +0.20 pp；
- 2/3 seed 为正；
- F1 相对 D3下降 <=0.10 pp。

---

# 14. 四轮方案复审

## Review 1：数据

确认 D1/D3 都是 3/3 F1 正方向；
QPRR 不再是假设，而是已获得 full-dataset 实验证据。

## Review 2：代码机制

定位出 QPRR-v1 的 same-class multi-GT pooling 与 cross-GT sorting 缺陷。

## Review 3：风险

否决直接扩大 TAL topk、修改 assignment、加入推理分支。
最终只做 GT-balanced training regularization。

## Review 4：实验稳定性

基于 D3 seed0 Recall 的明显偶然性，
把筛选最小单元从 1 seed 提高到 2 seeds。

---

# 15. 文献依据

- Rank & Sort Loss 支持“positive > negative”与按 localization quality 排序 classification score；
- TOOD/TAL 说明 classification-localization alignment 和 sample assignment 是 dense detection 的核心；
- CVPR 2023 One-to-Few 表明过少或不均衡 positives 会限制 representation learning；
- CVPR 2023 ARSL 指出 dense detector 中 selection/assignment ambiguity 会损害一阶段检测器；
- 最新 URPC2020 tiny-object 研究也报告提高困难目标感知往往带来 Recall/Precision trade-off，因此需要显式平衡 false negatives 与 false positives。

GBC-QPRR 并非照搬这些方法，而是针对本项目 QPRR-v1 的实际实验和代码行为做的最小修复。

---

# 16. 禁止事项

Codex 不得：

- 删除旧 qprr.py；
- 同时启用 qprr 与 gbc_qprr；
- 修改 TAL topk/alpha/beta；
- 修改 Detect；
- 修改 box/DFL；
- 修改 NMS；
- 修改 validation threshold；
- E1/E2 同时改变训练 recipe；
- 只跑 seed0 就宣布成功；
- 在 different-GT positives 间做 quality sorting；
- 用 classification score 选择 coverage positives。

---

# 17. 实验 manifest

```yaml
experiment: GBC-QPRR-YOLOv13
name: Ground-Truth-Balanced Coverage Precision-Recall Ranking
primary_dataset: /home/room305/ZZF/URPC2020/data.yaml
confirmatory_dataset: /home/room305/ZZF/URPC2020half

objective:
  accepted_metrics: [P, R, F1]
  primary_balanced_metric: F1
  f1_rule: compute_per_seed_then_mean_std

references:
  A0: reuse_existing_results
  A4: reuse_existing_results
  D1_A4_QPRR_v1: reuse_existing_results
  D3_A0_QPRR_v1: reuse_existing_results

runs:
  E0:
    train: false
    model: ultralytics/cfg/models/v13/yolov13n-gbc-qprr-e0-off.yaml
    purpose: exact A0 equivalence, gain=0
  E1:
    train: true
    model: ultralytics/cfg/models/v13/yolov13n-gbc-qprr-e1-instance.yaml
    purpose: isolate GT-balanced/intra-GT ranking; coverage disabled
  E2:
    train: true
    model: ultralytics/cfg/models/v13/yolov13n-gbc-qprr-e2-full.yaml
    purpose: main balanced candidate; GT-balanced ranking + coverage
  E3:
    train: conditional
    model: ultralytics/cfg/models/v13/yolov13n-a4-gbc-qprr-e3.yaml
    purpose: precision-oriented A4 base + full GBC-QPRR

fairness:
  inherit_exact_recipe: true
  disable_qprr_v1: true
  disable_sqanwd: true
  disable_f1r_head: true
  no_architecture_change_for_E1_E2: true
  do_not_change:
    - split
    - pretrained_checkpoint
    - epochs
    - imgsz
    - batch
    - optimizer
    - lr0
    - lrf
    - momentum
    - weight_decay
    - warmup
    - box
    - cls
    - dfl
    - augmentation
    - amp
    - workers
    - validation_thresholds

screening:
  note: "Do not use only seed0: D3 showed seed0 can be unrepresentative for Recall."
  initial_seeds: [0, 1]
  candidates: [E1, E2]
  advance_to_seed2_if:
    mean_F1_not_below_D3_first_two: true
    and_one_of:
      - mean_R_above_D3_first_two
      - mean_F1_above_D3_first_two
  E3_run_if: E2_passes_full_three_seed

acceptance:
  E2_vs_D3:
    mean_F1_gt_D3: true
    positive_F1_seeds_min: 2
    mean_R_gt_D3: true
    mean_P_not_below_A0: true
    AP_S_drop_vs_D3_pp_max: 0.20
    mAP50_95_drop_vs_D3_pp_max: 0.20
  paper_valid_alternative:
    precision_mode:
      mean_P_gain_vs_A0_pp_min: 0.30
      positive_P_seeds_min: 2
      F1_drop_vs_D3_pp_max: 0.10
    recall_mode:
      mean_R_gain_vs_A0_pp_min: 0.20
      positive_R_seeds_min: 2
      F1_drop_vs_D3_pp_max: 0.10

cross_dataset:
  run_half_after_full_pass: true
  require_same_primary_direction: true

```

---

# 18. 科学边界

当前可以比较有把握地说：

- QPRR 已经产生 P/F1 正增益；
- D3 是 3/3 seed F1 正方向；
- D1 是相对 A4 3/3 seed F1 正方向；
- 下一步继续围绕 QPRR 做 instance-level 修复，比重新发明一个 Neck 更有实验依据。

但真实 E1/E2/E3 是否进一步稳定提高 R/F1，
仍必须由训练结果验证，不能在训练前声称必然上涨。
