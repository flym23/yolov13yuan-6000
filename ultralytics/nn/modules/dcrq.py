"""DCRQ building blocks: bounded detail recovery and reliable cross-scale fusion."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .block import DSC3k2

__all__ = (
    "StrictResidualBudget",
    "DAWRB",
    "DAWRBDSC3k2",
    "RCFConcat",
    "RecallPreservingQualityCalibrator",
)


class StrictResidualBudget(nn.Module):
    """Apply a strict per-sample, per-channel RMS limit to a residual tensor."""

    def __init__(self, max_ratio=0.05, eps=1e-6, detach_scale=True):
        super().__init__()
        self.max_ratio = float(max_ratio)
        self.eps = float(eps)
        self.detach_scale = bool(detach_scale)
        if not 0.0 < self.max_ratio <= 1.0:
            raise ValueError(f"max_ratio must be in (0, 1], got {max_ratio}.")
        if self.eps <= 0.0:
            raise ValueError(f"eps must be positive, got {eps}.")

    def forward(self, base: torch.Tensor, correction: torch.Tensor) -> torch.Tensor:
        if base.shape != correction.shape:
            raise ValueError(f"base/correction shape mismatch: {tuple(base.shape)} vs {tuple(correction.shape)}.")
        if base.ndim != 4:
            raise ValueError("StrictResidualBudget expects BCHW tensors.")
        base_energy = base.float().square().mean(dim=(2, 3), keepdim=True).sqrt()
        correction_energy = correction.float().square().mean(dim=(2, 3), keepdim=True).add(self.eps).sqrt()
        scale = torch.minimum(
            torch.ones_like(correction_energy), self.max_ratio * base_energy / correction_energy.clamp_min(self.eps)
        )
        if self.detach_scale:
            scale = scale.detach()
        return (correction.float() * scale).to(dtype=base.dtype)


class DAWRB(nn.Module):
    """Degradation-Aware Wavelet Residual Booster inserted after backbone P3."""

    def __init__(self, channels, reduction=4, max_residual_ratio=0.05, detach_gate=True, eps=1e-6):
        super().__init__()
        self.channels = int(channels)
        self.reduction = int(reduction)
        self.detach_gate = bool(detach_gate)
        self.eps = float(eps)
        if self.channels <= 0:
            raise ValueError("channels must be positive.")
        if self.reduction < 1:
            raise ValueError("reduction must be >= 1.")
        if self.eps <= 0.0:
            raise ValueError("eps must be positive.")

        hidden = max(16, self.channels // self.reduction)
        self.detail_encoder = nn.Sequential(
            nn.Conv2d(3 * self.channels, 3 * self.channels, 3, 1, 1, groups=3 * self.channels, bias=False),
            nn.BatchNorm2d(3 * self.channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(3 * self.channels, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
        )
        self.reliability_gate = nn.Sequential(
            nn.Conv2d(4, 8, 1, bias=True), nn.SiLU(inplace=True), nn.Conv2d(8, 3, 1, bias=True), nn.Sigmoid()
        )
        self.band_projection = nn.Conv2d(hidden, 3 * self.channels, 1, bias=False)
        nn.init.zeros_(self.band_projection.weight)
        self.budget = StrictResidualBudget(max_ratio=max_residual_ratio, eps=self.eps, detach_scale=True)

    @staticmethod
    def _haar_dwt(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return Haar LL/LH/HL/HH bands, replicating odd trailing row or column."""
        height, width = x.shape[-2:]
        if height & 1 or width & 1:
            x = F.pad(x, (0, width & 1, 0, height & 1), mode="replicate")
        a, b = x[..., 0::2, 0::2], x[..., 0::2, 1::2]
        c, d = x[..., 1::2, 0::2], x[..., 1::2, 1::2]
        return (a + b + c + d) * 0.5, (-a - b + c + d) * 0.5, (-a + b - c + d) * 0.5, (a - b - c + d) * 0.5

    @staticmethod
    def _haar_idwt_detail(lh: torch.Tensor, hl: torch.Tensor, hh: torch.Tensor, out_hw: tuple[int, int]) -> torch.Tensor:
        """Invert only detail-band corrections; the low-frequency correction is exactly zero."""
        a, b = (-lh - hl + hh) * 0.5, (-lh + hl - hh) * 0.5
        c, d = (lh - hl - hh) * 0.5, (lh + hl + hh) * 0.5
        batch, channels, height, width = a.shape
        output = a.new_empty(batch, channels, height * 2, width * 2)
        output[..., 0::2, 0::2], output[..., 0::2, 1::2] = a, b
        output[..., 1::2, 0::2], output[..., 1::2, 1::2] = c, d
        return output[..., : out_hw[0], : out_hw[1]]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError(f"DAWRB expects BCHW with C={self.channels}, got {tuple(x.shape)}.")
        out_hw = x.shape[-2:]
        ll, lh, hl, hh = self._haar_dwt(x)
        detail_energy = torch.stack(
            (lh.float().abs().mean(dim=1), hl.float().abs().mean(dim=1), hh.float().abs().mean(dim=1)), dim=1
        )
        low_energy = ll.float().abs().mean(dim=1, keepdim=True)
        relative_detail = detail_energy / (low_energy + self.eps)
        normalized_low = low_energy / (low_energy.mean(dim=(2, 3), keepdim=True) + self.eps)
        gate = self.reliability_gate(torch.cat((relative_detail, normalized_low), dim=1).clamp(0.0, 8.0).to(x.dtype))
        if self.detach_gate:
            gate = gate.detach()
        delta_lh, delta_hl, delta_hh = self.band_projection(self.detail_encoder(torch.cat((lh, hl, hh), dim=1))).chunk(3, dim=1)
        correction = self._haar_idwt_detail(
            delta_lh * gate[:, 0:1], delta_hl * gate[:, 1:2], delta_hh * gate[:, 2:3], out_hw
        )
        return x + self.budget(x, correction)


class DAWRBDSC3k2(DSC3k2):
    """Same-index B3/B4 layer-4 replacement that keeps every inherited DSC3k2 checkpoint key."""

    def __init__(
        self,
        c1,
        c2,
        n=1,
        dsc3k=False,
        e=0.25,
        dawrb_reduction=4,
        dawrb_max_residual_ratio=0.05,
        dawrb_detach_gate=True,
        g=1,
        shortcut=True,
        k1=3,
        k2=7,
        d2=1,
        dawrb_eps=1e-6,
    ):
        # Build the inherited B2-compatible branch first, so model.4.cv*/m.* keys are unchanged.
        super().__init__(c1, c2, n, dsc3k, e, g, shortcut, k1, k2, d2)
        # New-only DAWRB state must not perturb initialization of downstream B2-compatible layers.
        with torch.random.fork_rng(devices=[], enabled=True):
            self.dawrb = DAWRB(c2, dawrb_reduction, dawrb_max_residual_ratio, dawrb_detach_gate, dawrb_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dawrb(super().forward(x))


class RCFConcat(nn.Module):
    """Reliability-Conflict Fusion Concat with an identity-initialized deep residual."""

    def __init__(self, c_deep, c_lateral, reduction=4, max_residual_ratio=0.04, detach_gate=True, eps=1e-6):
        super().__init__()
        self.c_deep, self.c_lateral = int(c_deep), int(c_lateral)
        self.reduction, self.detach_gate, self.eps = int(reduction), bool(detach_gate), float(eps)
        if self.c_deep <= 0 or self.c_lateral <= 0:
            raise ValueError("RCFConcat channel counts must be positive.")
        if self.reduction < 1 or self.eps <= 0.0:
            raise ValueError("reduction and eps must be positive.")
        hidden = max(16, min(self.c_deep, self.c_lateral) // self.reduction)
        # A plain Concat owns no randomly initialized state. Isolate this new branch to preserve later B2 RNG.
        with torch.random.fork_rng(devices=[], enabled=True):
            self.deep_proj = nn.Sequential(nn.Conv2d(self.c_deep, hidden, 1, bias=False), nn.BatchNorm2d(hidden), nn.SiLU(inplace=True))
            self.lateral_proj = nn.Sequential(nn.Conv2d(self.c_lateral, hidden, 1, bias=False), nn.BatchNorm2d(hidden), nn.SiLU(inplace=True))
            self.fusion = nn.Sequential(
                nn.Conv2d(3 * hidden, hidden, 3, 1, 1, bias=False), nn.BatchNorm2d(hidden), nn.SiLU(inplace=True),
                nn.Conv2d(hidden, hidden, 3, 1, 1, groups=hidden, bias=False), nn.BatchNorm2d(hidden), nn.SiLU(inplace=True),
            )
            self.permission = nn.Sequential(
                nn.Conv2d(3, 8, 1, bias=True), nn.SiLU(inplace=True), nn.Conv2d(8, 1, 1, bias=True), nn.Sigmoid()
            )
            self.out_projection = nn.Conv2d(hidden, self.c_deep, 1, bias=False)
        nn.init.zeros_(self.out_projection.weight)
        self.budget = StrictResidualBudget(max_ratio=max_residual_ratio, eps=self.eps, detach_scale=True)

    def forward(self, x: list[torch.Tensor] | tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise ValueError("RCFConcat expects [deep_feature, lateral_feature].")
        deep, lateral = x
        if deep.ndim != 4 or lateral.ndim != 4:
            raise ValueError("RCFConcat expects BCHW tensors.")
        if deep.shape[0] != lateral.shape[0] or deep.shape[-2:] != lateral.shape[-2:]:
            raise ValueError("RCFConcat deep/lateral batch and spatial dimensions must match.")
        if deep.shape[1] != self.c_deep or lateral.shape[1] != self.c_lateral:
            raise ValueError(f"RCFConcat expects channels {self.c_deep}/{self.c_lateral}.")
        if deep.device != lateral.device or deep.dtype != lateral.dtype:
            raise ValueError("RCFConcat deep and lateral tensors must share device and dtype.")
        deep_embed, lateral_embed = self.deep_proj(deep), self.lateral_proj(lateral)
        cosine = (F.normalize(deep_embed.float(), dim=1, eps=self.eps) * F.normalize(lateral_embed.float(), dim=1, eps=self.eps)).sum(
            dim=1, keepdim=True
        ).clamp(-1.0, 1.0)
        lateral_summary = lateral.float().abs().mean(dim=1, keepdim=True)
        local_mean = F.avg_pool2d(F.pad(lateral_summary, (1, 1, 1, 1), mode="replicate"), 3, 1)
        detail_ratio = ((lateral_summary - local_mean).abs() / (local_mean.abs() + self.eps)).clamp(0.0, 8.0)
        cues = torch.cat(((cosine + 1.0) * 0.5, (1.0 - cosine) * 0.5, 1.0 - torch.exp(-detail_ratio)), dim=1).to(deep.dtype)
        gate = self.permission(cues)
        if self.detach_gate:
            gate = gate.detach()
        correction = self.out_projection(self.fusion(torch.cat((deep_embed, lateral_embed, (deep_embed - lateral_embed).abs()), dim=1))) * gate
        return torch.cat((deep + self.budget(deep, correction), lateral), dim=1)


class RecallPreservingQualityCalibrator(nn.Module):
    """Boost confident quality estimates while applying only weak low-quality suppression."""

    def __init__(self, nl=3, strengths=(0.25, 0.20, 0.15), threshold=0.50, negative_ratio=0.25, max_factor=1.25):
        super().__init__()
        self.nl, self.threshold = int(nl), float(threshold)
        self.negative_ratio, self.max_factor = float(negative_ratio), float(max_factor)
        if self.nl < 1 or len(strengths) != self.nl:
            raise ValueError("nl must be positive and match the number of strengths.")
        if not 0.0 <= self.threshold <= 1.0 or not 0.0 <= self.negative_ratio <= 1.0 or self.max_factor < 1.0:
            raise ValueError("Invalid RPQC threshold, negative_ratio, or max_factor.")
        values = torch.tensor(strengths, dtype=torch.float32)
        if torch.any(values < 0):
            raise ValueError("RPQC strengths must be non-negative.")
        self.register_buffer("strengths", values, persistent=True)

    def forward(self, cls_prob: torch.Tensor, quality_prob: torch.Tensor, level_index: int) -> torch.Tensor:
        if cls_prob.ndim != 4 or quality_prob.ndim != 4 or quality_prob.shape[1] != 1:
            raise ValueError("RPQC expects class BCHW and single-channel quality BCHW tensors.")
        if cls_prob.shape[0] != quality_prob.shape[0] or cls_prob.shape[-2:] != quality_prob.shape[-2:]:
            raise ValueError("RPQC class/quality batch and spatial dimensions must match.")
        level_index = int(level_index)
        if not 0 <= level_index < self.nl:
            raise IndexError(f"RPQC level index {level_index} is outside [0, {self.nl}).")
        strength = self.strengths[level_index].to(device=cls_prob.device, dtype=cls_prob.dtype)
        factor = 1.0 + strength * F.relu(quality_prob - self.threshold) - strength * self.negative_ratio * F.relu(self.threshold - quality_prob)
        return (cls_prob * factor.clamp(0.0, self.max_factor)).clamp(0.0, 1.0)
