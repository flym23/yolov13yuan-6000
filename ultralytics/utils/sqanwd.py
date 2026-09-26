from __future__ import annotations

import torch
import torch.nn as nn


__all__ = ("ScaleQualityAdaptiveNWD",)


class ScaleQualityAdaptiveNWD(nn.Module):
    """
    Scale-Quality Adaptive Normalized Wasserstein Distance core.

    This module is deliberately independent of YOLO-specific BboxLoss/DFL code.
    It produces an elementwise regression loss that can be mixed with CIoU.

    Design goals
    ------------
    1. Resolution/aspect-ratio normalization:
       x quantities are normalized by image width and y quantities by image height.
    2. Small-object selectivity:
       NWD is used mainly for boxes whose normalized geometric-mean size is small.
    3. Quality self-annealing:
       NWD is progressively disabled as plain IoU becomes high, so CIoU owns
       high-quality localization.
    4. Mean calibration:
       NWD loss magnitude is calibrated to the current CIoU magnitude using
       detached statistics, then the final hybrid receives a bounded mean guard.
    5. Exact off switch:
       max_mix=0 returns the input CIoU loss exactly.
    """

    def __init__(
        self,
        max_mix: float = 0.20,
        c_ratio: float = 0.020,
        small_thr: float = 0.050,
        small_temp: float = 0.0125,
        iou_floor: float = 0.30,
        iou_ceiling: float = 0.80,
        calibration_min: float = 0.50,
        calibration_max: float = 2.00,
        mean_guard_min: float = 0.80,
        mean_guard_max: float = 1.25,
        eps: float = 1e-9,
    ):
        super().__init__()
        self.max_mix = float(max_mix)
        self.c_ratio = float(c_ratio)
        self.small_thr = float(small_thr)
        self.small_temp = float(small_temp)
        self.iou_floor = float(iou_floor)
        self.iou_ceiling = float(iou_ceiling)
        self.calibration_min = float(calibration_min)
        self.calibration_max = float(calibration_max)
        self.mean_guard_min = float(mean_guard_min)
        self.mean_guard_max = float(mean_guard_max)
        self.eps = float(eps)

        if not 0.0 <= self.max_mix <= 1.0:
            raise ValueError("max_mix must be in [0, 1]")
        if self.c_ratio <= 0.0:
            raise ValueError("c_ratio must be positive")
        if not 0.0 < self.small_thr < 1.0:
            raise ValueError("small_thr must be in (0, 1)")
        if self.small_temp <= 0.0:
            raise ValueError("small_temp must be positive")
        if not 0.0 <= self.iou_floor < self.iou_ceiling <= 1.0:
            raise ValueError("require 0 <= iou_floor < iou_ceiling <= 1")
        if not 0.0 < self.calibration_min <= self.calibration_max:
            raise ValueError("invalid calibration clamp")
        if not 0.0 < self.mean_guard_min <= self.mean_guard_max:
            raise ValueError("invalid mean-guard clamp")
        if self.eps <= 0.0:
            raise ValueError("eps must be positive")

    @staticmethod
    def _check_boxes(boxes: torch.Tensor, name: str):
        if boxes.ndim != 2 or boxes.shape[-1] != 4:
            raise ValueError(f"{name} must have shape [N, 4]")

    @staticmethod
    def _xyxy_to_cxcywh(boxes: torch.Tensor):
        x1, y1, x2, y2 = boxes.unbind(-1)
        w = (x2 - x1).clamp_min(0.0)
        h = (y2 - y1).clamp_min(0.0)
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        return cx, cy, w, h

    def nwd_similarity(
        self,
        pred_xyxy_px: torch.Tensor,
        target_xyxy_px: torch.Tensor,
        imgsz_hw: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return [N, 1] similarity.

        The normalized squared 2-Wasserstein box distance is:
            (dcx/W)^2 + (dcy/H)^2
            + 1/4 * ((dw/W)^2 + (dh/H)^2)

        This is invariant to independent x/y image resizing if boxes are resized
        by the same factors.
        """
        self._check_boxes(pred_xyxy_px, "pred_xyxy_px")
        self._check_boxes(target_xyxy_px, "target_xyxy_px")
        if pred_xyxy_px.shape != target_xyxy_px.shape:
            raise ValueError("pred/target boxes must share shape")
        if imgsz_hw.numel() != 2:
            raise ValueError("imgsz_hw must contain [H, W]")

        out_dtype = pred_xyxy_px.dtype
        pred = pred_xyxy_px.float()
        target = target_xyxy_px.float()
        hw = imgsz_hw.float().reshape(-1)
        h = hw[0].clamp_min(self.eps)
        w = hw[1].clamp_min(self.eps)

        pcx, pcy, pw, ph = self._xyxy_to_cxcywh(pred)
        tcx, tcy, tw, th = self._xyxy_to_cxcywh(target)

        d2 = (
            ((pcx - tcx) / w).square()
            + ((pcy - tcy) / h).square()
            + 0.25 * ((pw - tw) / w).square()
            + 0.25 * ((ph - th) / h).square()
        )

        # Exact matches should be exactly 1.0 rather than exp(-sqrt(eps)/c).
        sim = torch.exp(-torch.sqrt(d2.clamp_min(0.0)) / self.c_ratio)
        sim = torch.where(d2 <= self.eps, torch.ones_like(sim), sim)
        return sim.unsqueeze(-1).to(dtype=out_dtype)

    def size_gate(
        self,
        target_xyxy_px: torch.Tensor,
        imgsz_hw: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return smooth small-object gate and normalized geometric-mean size."""
        self._check_boxes(target_xyxy_px, "target_xyxy_px")
        out_dtype = target_xyxy_px.dtype
        target = target_xyxy_px.float()
        hw = imgsz_hw.float().reshape(-1)
        h_img = hw[0].clamp_min(self.eps)
        w_img = hw[1].clamp_min(self.eps)

        _, _, w, h = self._xyxy_to_cxcywh(target)
        size_ratio = torch.sqrt(
            ((w / w_img) * (h / h_img)).clamp_min(0.0)
        )
        gate = torch.sigmoid((self.small_thr - size_ratio) / self.small_temp)
        return gate.unsqueeze(-1).to(out_dtype), size_ratio.unsqueeze(-1).to(out_dtype)

    def quality_gate(self, plain_iou: torch.Tensor) -> torch.Tensor:
        """
        NWD is fully active below iou_floor, linearly annealed in the middle,
        and disabled at/above iou_ceiling.

        The gate is detached by design: it selects a loss regime but does not
        create an extra gradient path through IoU.
        """
        iou = plain_iou.detach().float().clamp(0.0, 1.0)
        gate = (self.iou_ceiling - iou) / (self.iou_ceiling - self.iou_floor)
        return gate.clamp(0.0, 1.0).to(dtype=plain_iou.dtype)

    def weighted_mean(
        self,
        value: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        if value.shape != weight.shape:
            raise ValueError(
                f"value/weight shape mismatch: {value.shape} vs {weight.shape}"
            )
        denom = weight.float().sum().clamp_min(self.eps)
        return (value.float() * weight.float()).sum() / denom

    def forward(
        self,
        ciou_loss: torch.Tensor,
        plain_iou: torch.Tensor,
        pred_xyxy_px: torch.Tensor,
        target_xyxy_px: torch.Tensor,
        weight: torch.Tensor,
        imgsz_hw: torch.Tensor,
    ):
        if ciou_loss.shape != plain_iou.shape or ciou_loss.shape != weight.shape:
            raise ValueError("ciou_loss/plain_iou/weight must share shape [N, 1]")
        if ciou_loss.ndim != 2 or ciou_loss.shape[-1] != 1:
            raise ValueError("loss tensors must have shape [N, 1]")

        # Exact baseline mode, useful both for tests and ablation.
        if self.max_mix == 0.0:
            ones = torch.ones_like(ciou_loss)
            zeros = torch.zeros_like(ciou_loss)
            return ciou_loss, {
                "nwd_similarity": ones,
                "nwd_loss": zeros,
                "small_gate": zeros,
                "quality_gate": zeros,
                "mix": zeros,
                "nwd_calibration": ciou_loss.new_tensor(1.0),
                "mean_guard": ciou_loss.new_tensor(1.0),
            }

        nwd_similarity = self.nwd_similarity(
            pred_xyxy_px, target_xyxy_px, imgsz_hw
        )
        nwd_loss = 1.0 - nwd_similarity

        small_gate, size_ratio = self.size_gate(target_xyxy_px, imgsz_hw)
        q_gate = self.quality_gate(plain_iou)
        mix = self.max_mix * small_gate * q_gate

        ciou_mean = self.weighted_mean(ciou_loss, weight)
        nwd_mean = self.weighted_mean(nwd_loss, weight)

        nwd_calibration = (
            ciou_mean.detach() / nwd_mean.detach().clamp_min(self.eps)
        ).clamp(self.calibration_min, self.calibration_max)
        nwd_loss_cal = nwd_loss * nwd_calibration.to(nwd_loss.dtype)

        hybrid_pre = (1.0 - mix) * ciou_loss + mix * nwd_loss_cal

        hybrid_mean = self.weighted_mean(hybrid_pre, weight)
        mean_guard = (
            ciou_mean.detach() / hybrid_mean.detach().clamp_min(self.eps)
        ).clamp(self.mean_guard_min, self.mean_guard_max)

        hybrid = hybrid_pre * mean_guard.to(hybrid_pre.dtype)

        return hybrid, {
            "nwd_similarity": nwd_similarity,
            "nwd_loss": nwd_loss,
            "size_ratio": size_ratio,
            "small_gate": small_gate,
            "quality_gate": q_gate,
            "mix": mix,
            "nwd_calibration": nwd_calibration,
            "mean_guard": mean_guard,
        }
