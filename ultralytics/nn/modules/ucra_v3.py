from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ("UCRA1v3", "UCRA2v3")


class _ConvGNAct(nn.Module):
    """Small-batch-friendly projection used only inside the auxiliary branch."""

    def __init__(self, c1, c2, k=1, s=1, groups=1, act=True, gn_groups=8):
        super().__init__()
        c1, c2, k, s, groups = int(c1), int(c2), int(k), int(s), int(groups)
        if min(c1, c2, k, s, groups) <= 0:
            raise ValueError("invalid convolution arguments")
        if c1 % groups != 0 or c2 % groups != 0:
            raise ValueError(f"groups={groups} must divide c1={c1} and c2={c2}")

        self.conv = nn.Conv2d(c1, c2, k, s, k // 2, groups=groups, bias=False)
        ng = min(int(gn_groups), c2)
        while ng > 1 and c2 % ng:
            ng -= 1
        self.norm = nn.GroupNorm(ng, c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class _ResidualBudget(nn.Module):
    """Hard RMS limiter. A zero-energy main path receives exactly zero correction."""

    def __init__(self, max_ratio, per_channel=False, detach_scale=True, eps=1e-6):
        super().__init__()
        self.max_ratio = float(max_ratio)
        self.per_channel = bool(per_channel)
        self.detach_scale = bool(detach_scale)
        self.eps = float(eps)
        if not 0.0 < self.max_ratio <= 1.0:
            raise ValueError("max_ratio must be in (0, 1]")
        if self.eps <= 0.0:
            raise ValueError("eps must be positive")

    def forward(self, base, correction):
        if base.shape != correction.shape:
            raise ValueError("base and correction shapes differ")

        dims = (2, 3) if self.per_channel else (1, 2, 3)
        base_rms = base.float().square().mean(dim=dims, keepdim=True).sqrt()
        corr_rms = correction.float().square().mean(dim=dims, keepdim=True).add(self.eps).sqrt()

        scale = torch.minimum(
            torch.ones_like(corr_rms),
            self.max_ratio * base_rms / corr_rms.clamp_min(self.eps),
        )
        if self.detach_scale:
            scale = scale.detach()
        return (correction.float() * scale).to(base.dtype), scale


class _UCRACommon(nn.Module):
    """Common analytic reliability and multi-band utilities for UCRA-v3."""

    def __init__(
        self,
        c_deep,
        c_lateral,
        scale=2,
        reduction=4,
        strict_scale=True,
        detach_analytic=True,
        eps=1e-6,
    ):
        super().__init__()
        self.c_deep = int(c_deep)
        self.c_lateral = int(c_lateral)
        self.scale = int(scale)
        self.strict_scale = bool(strict_scale)
        self.detach_analytic = bool(detach_analytic)
        self.eps = float(eps)
        reduction = int(reduction)

        if min(self.c_deep, self.c_lateral) <= 0:
            raise ValueError("channel counts must be positive")
        if self.scale <= 1 or reduction < 1 or self.eps <= 0.0:
            raise ValueError("invalid scale/reduction/eps")

        self.hidden = max(16, min(64, min(self.c_deep, self.c_lateral) // reduction))

        kernel3 = torch.tensor(
            ((1.0, 2.0, 1.0), (2.0, 4.0, 2.0), (1.0, 2.0, 1.0)),
            dtype=torch.float32,
        )
        vector5 = torch.tensor((1.0, 4.0, 6.0, 4.0, 1.0), dtype=torch.float32)
        kernel5 = torch.outer(vector5, vector5)
        sobel_x = torch.tensor(
            ((-1.0, 0.0, 1.0), (-2.0, 0.0, 2.0), (-1.0, 0.0, 1.0)),
            dtype=torch.float32,
        ) / 8.0

        self.register_buffer("blur3_kernel", (kernel3 / kernel3.sum())[None, None], persistent=False)
        self.register_buffer("blur5_kernel", (kernel5 / kernel5.sum())[None, None], persistent=False)
        self.register_buffer("sobel_x", sobel_x[None, None], persistent=False)
        self.register_buffer("sobel_y", sobel_x.t().contiguous()[None, None], persistent=False)

        self.deep_proj = _ConvGNAct(self.c_deep, self.hidden, 1, 1)
        self.lateral_proj = _ConvGNAct(self.c_lateral, self.hidden, 1, 1)

    def _validate(self, deep, lateral):
        if deep.ndim != 4 or lateral.ndim != 4:
            raise ValueError("UCRA-v3 expects NCHW tensors")
        if deep.shape[0] != lateral.shape[0]:
            raise ValueError("deep/lateral batch sizes differ")
        if deep.shape[1] != self.c_deep or lateral.shape[1] != self.c_lateral:
            raise ValueError(
                f"expected channels {self.c_deep}/{self.c_lateral}, "
                f"got {deep.shape[1]}/{lateral.shape[1]}"
            )
        if deep.device != lateral.device or deep.dtype != lateral.dtype:
            raise ValueError("deep/lateral must share device and dtype")

        expected = (deep.shape[-2] * self.scale, deep.shape[-1] * self.scale)
        if self.strict_scale and tuple(lateral.shape[-2:]) != expected:
            raise ValueError(f"expected lateral size {expected}, got {tuple(lateral.shape[-2:])}")

    @staticmethod
    def _zscore(x, eps):
        value = x.float()
        mean = value.mean(dim=(2, 3), keepdim=True)
        std = value.var(dim=(2, 3), keepdim=True, unbiased=False).add(eps).sqrt()
        return (value - mean) / std

    def _blur(self, x, kernel_size):
        if kernel_size == 3:
            kernel = self.blur3_kernel
        elif kernel_size == 5:
            kernel = self.blur5_kernel
        else:
            raise ValueError("kernel_size must be 3 or 5")

        radius = kernel.shape[-1] // 2
        weight = kernel.to(device=x.device, dtype=x.dtype).repeat(x.shape[1], 1, 1, 1)
        return F.conv2d(
            F.pad(x, (radius, radius, radius, radius), mode="replicate"),
            weight,
            groups=x.shape[1],
        )

    def _semantic_reliability(self, deep_embed, lateral_embed):
        """Direction + amplitude consensus. Both terms are required."""
        deep_low = self._blur(deep_embed, 5)
        lateral_low = self._blur(lateral_embed, 5)

        deep_norm = F.normalize(deep_low.float(), dim=1, eps=self.eps)
        lateral_norm = F.normalize(lateral_low.float(), dim=1, eps=self.eps)
        agreement = (
            (deep_norm * lateral_norm).sum(dim=1, keepdim=True).add(1.0).mul(0.5).clamp(0.0, 1.0)
        )

        deep_energy = deep_low.float().abs().mean(dim=1, keepdim=True)
        lateral_energy = lateral_low.float().abs().mean(dim=1, keepdim=True)
        discrepancy = (deep_low.float() - lateral_low.float()).abs().mean(dim=1, keepdim=True)
        consistency = torch.exp(
            -(discrepancy / (deep_energy + lateral_energy + self.eps)).clamp(0.0, 8.0)
        )

        semantic = torch.sqrt((agreement * consistency).clamp_min(0.0) + self.eps).clamp(0.0, 1.0)
        return semantic, agreement, consistency, deep_low, lateral_low

    def _structure_support(self, lateral_embed, lateral_low5):
        """Separate mid-band structure from noise-prone highest-frequency detail."""
        low3 = self._blur(lateral_embed, 3)
        mid = low3 - lateral_low5
        high = lateral_embed - low3

        low_energy = lateral_low5.float().abs().mean(dim=1, keepdim=True)
        mid_energy = mid.float().abs().mean(dim=1, keepdim=True)
        high_energy = high.float().abs().mean(dim=1, keepdim=True)

        mid_absolute = 1.0 - torch.exp(-(mid_energy / (low_energy + self.eps)).clamp(0.0, 8.0))
        mid_relative = torch.sigmoid(self._zscore(mid_energy, self.eps))
        mid_support = (mid_absolute * mid_relative).clamp(0.0, 1.0)

        summary = lateral_embed.float().square().mean(dim=1, keepdim=True).add(self.eps).sqrt()
        padded = F.pad(summary, (1, 1, 1, 1), mode="replicate")
        grad_x = F.conv2d(padded, self.sobel_x.to(device=summary.device, dtype=summary.dtype))
        grad_y = F.conv2d(padded, self.sobel_y.to(device=summary.device, dtype=summary.dtype))

        jxx = F.avg_pool2d(F.pad(grad_x.square(), (1, 1, 1, 1), mode="replicate"), 3, 1)
        jyy = F.avg_pool2d(F.pad(grad_y.square(), (1, 1, 1, 1), mode="replicate"), 3, 1)
        jxy = F.avg_pool2d(F.pad(grad_x * grad_y, (1, 1, 1, 1), mode="replicate"), 3, 1)
        coherence = (
            ((jxx - jyy).square() + 4.0 * jxy.square() + self.eps).sqrt()
            / (jxx + jyy + self.eps)
        ).clamp(0.0, 1.0)

        high_absolute = 1.0 - torch.exp(
            -(high_energy / (low_energy + mid_energy + self.eps)).clamp(0.0, 8.0)
        )
        high_relative = torch.sigmoid(self._zscore(high_energy, self.eps))
        high_support = (high_absolute * high_relative * coherence).clamp(0.0, 1.0)

        structure = (0.65 * mid_support + 0.35 * high_support).clamp(0.0, 1.0)
        return structure, coherence, mid, high


class UCRA1v3(_UCRACommon):
    """P5->P4: conservative semantic transport, no learned spatial displacement."""

    def __init__(
        self,
        c_deep,
        c_lateral,
        scale=2,
        reduction=4,
        max_residual_ratio=0.06,
        reliability_floor=0.10,
        strict_scale=True,
        detach_analytic=True,
        detach_budget=True,
        eps=1e-6,
    ):
        # Replacing parameter-free Upsample must not perturb downstream RNG initialization.
        with torch.random.fork_rng(devices=[], enabled=True):
            super().__init__(
                c_deep,
                c_lateral,
                scale=scale,
                reduction=reduction,
                strict_scale=strict_scale,
                detach_analytic=detach_analytic,
                eps=eps,
            )
            self.reliability_floor = float(reliability_floor)
            if not 0.0 <= self.reliability_floor <= 1.0:
                raise ValueError("reliability_floor must be in [0, 1]")

            self.semantic_out = nn.Conv2d(self.hidden, self.c_deep, 1, bias=False)
            nn.init.zeros_(self.semantic_out.weight)

        self.budget = _ResidualBudget(
            max_residual_ratio,
            per_channel=True,
            detach_scale=detach_budget,
            eps=eps,
        )
        self.record_diagnostics = False
        self.latest_diagnostics = {}

    def set_diagnostics(self, enabled=True):
        self.record_diagnostics = bool(enabled)
        if not self.record_diagnostics:
            self.latest_diagnostics = {}
        return self

    def compute_components(self, deep, lateral):
        self._validate(deep, lateral)
        target_size = tuple(lateral.shape[-2:])
        base = F.interpolate(deep, size=target_size, mode="nearest")

        deep_embed = F.interpolate(
            self.deep_proj(deep),
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
        lateral_embed = self.lateral_proj(lateral)

        semantic, agreement, consistency, deep_low, lateral_low = self._semantic_reliability(
            deep_embed, lateral_embed
        )
        permission = self.reliability_floor + (1.0 - self.reliability_floor) * semantic

        if self.detach_analytic:
            semantic = semantic.detach()
            agreement = agreement.detach()
            consistency = consistency.detach()
            permission = permission.detach()

        candidate = permission.to(deep.dtype) * (lateral_low - deep_low)
        raw_correction = self.semantic_out(candidate)
        correction, budget_scale = self.budget(base, raw_correction)

        return {
            "base": base,
            "correction": correction,
            "semantic": semantic,
            "agreement": agreement,
            "consistency": consistency,
            "permission": permission,
            "budget_scale": budget_scale,
        }

    def forward(self, inputs):
        if not isinstance(inputs, (list, tuple)) or len(inputs) != 2:
            raise TypeError("UCRA1v3 expects [P5_deep, P4_lateral]")

        parts = self.compute_components(inputs[0], inputs[1])
        output = parts["base"] + parts["correction"]

        if self.record_diagnostics:
            self.latest_diagnostics = {
                key: value.detach() for key, value in parts.items() if key != "base"
            }
        return output


class UCRA2v3(_UCRACommon):
    """P4->P3: consensus-stabilized central-difference resampling and multi-band detail release."""

    def __init__(
        self,
        c_deep,
        c_lateral,
        scale=2,
        reduction=4,
        sample_groups=4,
        max_offset=0.35,
        max_geom_gain=0.10,
        max_residual_ratio=0.08,
        reliability_floor=0.08,
        strict_scale=True,
        detach_analytic=True,
        detach_budget=True,
        eps=1e-6,
    ):
        with torch.random.fork_rng(devices=[], enabled=True):
            super().__init__(
                c_deep,
                c_lateral,
                scale=scale,
                reduction=reduction,
                strict_scale=strict_scale,
                detach_analytic=detach_analytic,
                eps=eps,
            )
            requested_groups = int(sample_groups)
            if requested_groups < 1:
                raise ValueError("sample_groups must be positive")
            self.sample_groups = max(1, math.gcd(self.c_deep, requested_groups))
            self.group_channels = self.c_deep // self.sample_groups

            self.max_offset = float(max_offset)
            self.max_geom_gain = float(max_geom_gain)
            self.reliability_floor = float(reliability_floor)
            if not 0.0 <= self.max_offset <= 1.0:
                raise ValueError("max_offset must be in [0, 1]")
            if not 0.0 <= self.max_geom_gain <= 1.0:
                raise ValueError("max_geom_gain must be in [0, 1]")
            if not 0.0 <= self.reliability_floor <= 1.0:
                raise ValueError("reliability_floor must be in [0, 1]")

            self.offset_head = nn.Sequential(
                _ConvGNAct(3 * self.hidden + 2, self.hidden, 3, 1),
                _ConvGNAct(self.hidden, self.hidden, 3, 1, groups=self.hidden),
                nn.Conv2d(self.hidden, 2 * self.sample_groups, 1, bias=True),
            )
            # Offsets are non-zero but very small at init. The geometric gain is exactly
            # zero, so the whole module remains an exact baseline mapping while alpha can
            # receive a first-step gradient.
            nn.init.normal_(self.offset_head[-1].weight, mean=0.0, std=1e-1)
            nn.init.zeros_(self.offset_head[-1].bias)

            self.detail_out = nn.Conv2d(self.hidden, self.c_deep, 1, bias=False)
            nn.init.zeros_(self.detail_out.weight)

            self.geom_gain_raw = nn.Parameter(torch.zeros(1, dtype=torch.float32))

        self.budget = _ResidualBudget(
            max_residual_ratio,
            per_channel=False,
            detach_scale=detach_budget,
            eps=eps,
        )
        self.record_diagnostics = False
        self.latest_diagnostics = {}

    def set_diagnostics(self, enabled=True):
        self.record_diagnostics = bool(enabled)
        if not self.record_diagnostics:
            self.latest_diagnostics = {}
        return self

    @staticmethod
    def _base_grid(batch, height, width, device):
        y = (torch.arange(height, device=device, dtype=torch.float32) + 0.5) * (
            2.0 / max(height, 1)
        ) - 1.0
        x = (torch.arange(width, device=device, dtype=torch.float32) + 0.5) * (
            2.0 / max(width, 1)
        ) - 1.0
        gy, gx = torch.meshgrid(y, x, indexing="ij")
        return torch.stack((gx, gy), dim=-1).unsqueeze(0).expand(batch, -1, -1, -1)

    def _sample(self, deep, offsets, out_size, sign):
        batch, channels, in_h, in_w = deep.shape
        out_h, out_w = out_size
        groups = self.sample_groups

        expected = (batch, 2 * groups, out_h, out_w)
        if tuple(offsets.shape) != expected:
            raise ValueError(f"unexpected offset shape {tuple(offsets.shape)}, expected {expected}")

        offset_view = offsets.view(batch, groups, 2, out_h, out_w).float()
        offset_x = float(sign) * offset_view[:, :, 0] * (2.0 / max(in_w, 1))
        offset_y = float(sign) * offset_view[:, :, 1] * (2.0 / max(in_h, 1))
        delta_grid = torch.stack((offset_x, offset_y), dim=-1)

        grid = self._base_grid(batch, out_h, out_w, deep.device).unsqueeze(1) + delta_grid
        grid = grid.reshape(batch * groups, out_h, out_w, 2).to(deep.dtype)

        grouped = deep.reshape(
            batch, groups, self.group_channels, in_h, in_w
        ).reshape(batch * groups, self.group_channels, in_h, in_w)

        sampled = F.grid_sample(
            grouped,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        return sampled.reshape(
            batch, groups, self.group_channels, out_h, out_w
        ).reshape(batch, channels, out_h, out_w)

    @staticmethod
    def _local_mean(x):
        return F.avg_pool2d(F.pad(x, (1, 1, 1, 1), mode="replicate"), 3, 1)

    def _orthogonalize(self, base, correction):
        base_float = base.float()
        correction_float = correction.float()
        denominator = base_float.square().sum(dim=1, keepdim=True)
        projection = (
            (base_float * correction_float).sum(dim=1, keepdim=True)
            / (denominator + self.eps)
        )
        return (correction_float - projection * base_float).to(correction.dtype)

    def compute_components(self, deep, lateral):
        self._validate(deep, lateral)
        target_size = tuple(lateral.shape[-2:])
        base = F.interpolate(deep, size=target_size, mode="nearest")

        deep_embed = F.interpolate(
            self.deep_proj(deep),
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
        lateral_embed = self.lateral_proj(lateral)

        semantic, agreement, consistency, _, lateral_low5 = self._semantic_reliability(
            deep_embed, lateral_embed
        )
        structure, coherence, mid_band, high_band = self._structure_support(
            lateral_embed, lateral_low5
        )

        # Unlike UCRA-v2's weighted sum, semantic support is mandatory here.
        # Strong high-frequency texture alone cannot fully open the release gate.
        consensus = semantic * (0.35 + 0.65 * structure)
        permission = self.reliability_floor + (1.0 - self.reliability_floor) * consensus
        geometry_permission = semantic * (0.20 + 0.80 * structure)

        if self.detach_analytic:
            semantic = semantic.detach()
            agreement = agreement.detach()
            consistency = consistency.detach()
            structure = structure.detach()
            coherence = coherence.detach()
            permission = permission.detach()
            geometry_permission = geometry_permission.detach()

        offset_cues = torch.cat(
            (
                deep_embed,
                lateral_embed,
                (deep_embed - lateral_embed).abs(),
                semantic.to(deep.dtype),
                structure.to(deep.dtype),
            ),
            dim=1,
        )
        raw_offsets = self.offset_head(offset_cues)
        offset_permission = geometry_permission.repeat(
            1, 2 * self.sample_groups, 1, 1
        )
        offsets = self.max_offset * torch.tanh(raw_offsets) * offset_permission

        # Pure differential geometry:
        # delta_geo = 0.5 * [F(x + o) - F(x - o)].
        # Therefore max_offset=0 gives an exactly zero geometric source.
        # This fixes UCRA-v2 A3, where zero offset still contained a
        # bilinear-vs-nearest interpolation residual.
        plus = self._sample(deep, offsets, target_size, sign=+1.0)
        minus = self._sample(deep, offsets, target_size, sign=-1.0)
        geometry_source = 0.5 * (plus - minus)
        geometry_gain = (
            self.max_geom_gain * torch.tanh(self.geom_gain_raw)
        ).to(deep.dtype)
        geometry_correction = geometry_gain * geometry_source

        # Mid-band structure is always preferred. The highest-frequency band is
        # released only where orientation coherence and semantic reliability agree.
        high_keep = (coherence * semantic).clamp(0.0, 1.0).to(deep.dtype)
        detail_hidden = mid_band + high_keep * high_band
        detail_source = permission.to(deep.dtype) * detail_hidden
        detail_correction = self.detail_out(detail_source)

        raw_correction = geometry_correction + detail_correction

        # Precision protection: remove hidden low-frequency gain and prevent the
        # auxiliary branch from merely rescaling the existing nearest main path.
        raw_correction = raw_correction - self._local_mean(
            raw_correction.float()
        ).to(raw_correction.dtype)
        raw_correction = self._orthogonalize(base, raw_correction)

        correction, budget_scale = self.budget(base, raw_correction)

        return {
            "base": base,
            "correction": correction,
            "semantic": semantic,
            "agreement": agreement,
            "consistency": consistency,
            "structure": structure,
            "coherence": coherence,
            "permission": permission,
            "geometry_permission": geometry_permission,
            "offsets": offsets,
            "geometry_source": geometry_source,
            "geometry_gain": geometry_gain,
            "budget_scale": budget_scale,
        }

    def forward(self, inputs):
        if not isinstance(inputs, (list, tuple)) or len(inputs) != 2:
            raise TypeError("UCRA2v3 expects [P4_deep, P3_lateral]")

        parts = self.compute_components(inputs[0], inputs[1])
        output = parts["base"] + parts["correction"]

        if self.record_diagnostics:
            self.latest_diagnostics = {
                key: value.detach() for key, value in parts.items() if key != "base"
            }
        return output
