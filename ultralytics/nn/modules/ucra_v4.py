from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ucra_v2 import UCRA2v2

__all__ = ("UCRA2v4",)


class _PrecisionResidualBudget(nn.Module):
    """Sample-wise RMS limiter for the precision-only residual branch."""

    def __init__(self, max_ratio: float = 0.04, detach_scale: bool = True, eps: float = 1e-6):
        super().__init__()
        self.max_ratio = float(max_ratio)
        self.detach_scale = bool(detach_scale)
        self.eps = float(eps)
        if not 0.0 < self.max_ratio <= 1.0:
            raise ValueError("max_ratio must be in (0, 1]")
        if self.eps <= 0.0:
            raise ValueError("eps must be positive")

    def forward(self, reference: torch.Tensor, correction: torch.Tensor):
        if reference.shape != correction.shape:
            raise ValueError("reference and correction must share shape")
        ref_rms = reference.float().square().mean(dim=(1, 2, 3), keepdim=True).sqrt()
        corr_rms = correction.float().square().mean(dim=(1, 2, 3), keepdim=True).add(self.eps).sqrt()
        scale = torch.minimum(
            torch.ones_like(corr_rms),
            self.max_ratio * ref_rms / corr_rms.clamp_min(self.eps),
        )
        if self.detach_scale:
            scale = scale.detach()
        return (correction.float() * scale).to(reference.dtype), scale


def _init_grouped_identity_1x1(conv: nn.Conv2d):
    """Initialize a grouped 1x1 conv as exact channel identity."""
    if conv.kernel_size != (1, 1) or conv.in_channels != conv.out_channels:
        raise ValueError("grouped identity requires square 1x1 convolution")
    with torch.no_grad():
        conv.weight.zero_()
        channels_per_group = conv.in_channels // conv.groups
        for out_channel in range(conv.out_channels):
            local_input = out_channel % channels_per_group
            conv.weight[out_channel, local_input, 0, 0] = 1.0
        if conv.bias is not None:
            conv.bias.zero_()


class UCRA2v4(UCRA2v2):
    """P4->P3 UCRA-v4: A4 core + gradient-isolated sparse precision residual.

    Data-driven rationale:
      * UCRA2v2/A4 is retained intact because it gives the best AP_S/recall behavior.
      * A separate symmetric central-difference adapter targets the B2-observed gain in
        precision/high-IoU localization without replacing the A4 branch.

    The precision branch uses detached A4 reliability cues and detached source features,
    so it cannot directly rewrite the validated UCRA2v2 offset/reliability path. Its
    output is zero at initialization through a zero scalar gain, is spatially sparse,
    orthogonal to the A4 output, and is hard-bounded in RMS energy.
    """

    def __init__(
        self,
        c_deep: int,
        c_lateral: int,
        scale: int = 2,
        reduction: int = 4,
        sample_groups: int = 4,
        max_offset: float = 0.50,
        max_residual_ratio: float = 0.10,
        reliability_floor: float = 0.15,
        strict_scale: bool = True,
        detach_reliability: bool = True,
        detach_budget: bool = True,
        eps: float = 1e-6,
        max_precision_offset: float = 0.35,
        max_precision_gain: float = 0.15,
        max_precision_ratio: float = 0.04,
        precision_spatial_rho: float = 0.15,
        precision_groups: int = 4,
        detach_precision_source: bool = True,
    ):
        super().__init__(
            c_deep=c_deep,
            c_lateral=c_lateral,
            scale=scale,
            reduction=reduction,
            sample_groups=sample_groups,
            max_offset=max_offset,
            max_residual_ratio=max_residual_ratio,
            reliability_floor=reliability_floor,
            strict_scale=strict_scale,
            detach_reliability=detach_reliability,
            detach_budget=detach_budget,
            eps=eps,
        )

        self.max_precision_offset = float(max_precision_offset)
        self.max_precision_gain = float(max_precision_gain)
        self.max_precision_ratio = float(max_precision_ratio)
        self.precision_spatial_rho = float(precision_spatial_rho)
        self.detach_precision_source = bool(detach_precision_source)

        precision_groups = int(precision_groups)
        if not 0.0 <= self.max_precision_offset <= 1.0:
            raise ValueError("max_precision_offset must be in [0, 1]")
        if not 0.0 <= self.max_precision_gain <= 1.0:
            raise ValueError("max_precision_gain must be in [0, 1]")
        if not 0.0 < self.max_precision_ratio <= 1.0:
            raise ValueError("max_precision_ratio must be in (0, 1]")
        if not 0.0 < self.precision_spatial_rho <= 1.0:
            raise ValueError("precision_spatial_rho must be in (0, 1]")
        if precision_groups < 1:
            raise ValueError("precision_groups must be positive")

        self.precision_groups = max(1, math.gcd(self.c_deep, precision_groups))

        # New branch construction is RNG-isolated so downstream YOLO layers keep the
        # same initialization sequence. Offset logits start small but non-zero; because
        # precision_gain_raw==0, model output is still EXACTLY A4 at initialization.
        with torch.random.fork_rng(devices=[], enabled=True):
            self.precision_offset_head = nn.Sequential(
                nn.Conv2d(3 * self.hidden, 3 * self.hidden, 3, 1, 1, groups=3 * self.hidden, bias=False),
                nn.SiLU(inplace=True),
                nn.Conv2d(3 * self.hidden, self.hidden, 1, bias=False),
                nn.SiLU(inplace=True),
                nn.Conv2d(self.hidden, 2 * self.sample_groups, 1, bias=True),
            )
            nn.init.normal_(self.precision_offset_head[-1].weight, mean=0.0, std=1e-1)
            nn.init.zeros_(self.precision_offset_head[-1].bias)

            self.precision_out = nn.Conv2d(
                self.c_deep,
                self.c_deep,
                1,
                groups=self.precision_groups,
                bias=False,
            )
            _init_grouped_identity_1x1(self.precision_out)

        self.precision_gain_raw = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        self.precision_budget = _PrecisionResidualBudget(
            max_ratio=self.max_precision_ratio,
            detach_scale=detach_budget,
            eps=self.eps,
        )

    def _precision_spatial_budget(self, gate: torch.Tensor):
        if gate.ndim != 4 or gate.shape[1] != 1:
            raise ValueError("precision gate must have shape [B,1,H,W]")
        mean = gate.float().mean(dim=(2, 3), keepdim=True)
        scale = torch.minimum(
            torch.ones_like(mean),
            torch.full_like(mean, self.precision_spatial_rho) / mean.clamp_min(self.eps),
        ).detach()
        return (gate.float() * scale).clamp(0.0, 1.0).to(gate.dtype), scale

    def _precision_gate(self, parts: dict[str, torch.Tensor]):
        agreement = parts["agreement"].float().clamp(0.0, 1.0)
        consistency = parts["consistency"].float().clamp(0.0, 1.0)
        detail = parts["detail_reliability"].float().clamp(0.0, 1.0)

        # Multiplicative consensus: detail cannot act without cross-scale semantic support.
        semantic = torch.sqrt((agreement * consistency).clamp_min(0.0))
        gate = (semantic * detail).clamp(0.0, 1.0)
        gate, spatial_scale = self._precision_spatial_budget(gate)
        return gate, semantic, spatial_scale

    def _precision_cues(self, deep: torch.Tensor, lateral: torch.Tensor):
        target_hw = tuple(lateral.shape[-2:])
        if self.detach_precision_source:
            # Full gradient isolation from the A4 projections and backbone source.
            with torch.no_grad():
                deep_embed = F.interpolate(
                    self.deep_proj(deep),
                    size=target_hw,
                    mode="bilinear",
                    align_corners=False,
                )
                lateral_embed = self.lateral_proj(lateral)
        else:
            deep_embed = F.interpolate(
                self.deep_proj(deep),
                size=target_hw,
                mode="bilinear",
                align_corners=False,
            )
            lateral_embed = self.lateral_proj(lateral)

        cue = torch.cat((deep_embed, lateral_embed, (deep_embed - lateral_embed).abs()), dim=1)
        return cue.detach() if self.detach_precision_source else cue

    def compute_precision_components(
        self,
        deep: torch.Tensor,
        lateral: torch.Tensor,
        core_output: torch.Tensor,
        parts: dict[str, torch.Tensor],
    ):
        gate, semantic, spatial_scale = self._precision_gate(parts)
        cue = self._precision_cues(deep, lateral)
        raw_offsets = self.precision_offset_head(cue)

        gate_for_offset = gate.repeat(1, 2 * self.sample_groups, 1, 1)
        offsets = self.max_precision_offset * torch.tanh(raw_offsets) * gate_for_offset

        source_deep = deep.detach() if self.detach_precision_source else deep
        reference = core_output.detach() if self.detach_precision_source else core_output
        target_hw = tuple(core_output.shape[-2:])

        sampled_pos = self._dynamic_sample(source_deep, offsets, target_hw)
        sampled_neg = self._dynamic_sample(source_deep, -offsets, target_hw)

        # Pure odd component. No bilinear-minus-nearest term is present.
        geometry = 0.5 * (sampled_pos - sampled_neg)
        candidate = gate.to(geometry.dtype) * geometry

        raw = self.precision_out(candidate)
        raw = raw - self._local_mean(raw.float()).to(raw.dtype)
        raw = self._orthogonalize(reference, raw)

        gain = (self.max_precision_gain * torch.tanh(self.precision_gain_raw)).to(raw.dtype)
        gained = gain * raw
        correction, budget_scale = self.precision_budget(reference, gained)

        return {
            "correction": correction,
            "gate": gate,
            "semantic": semantic,
            "spatial_scale": spatial_scale,
            "offsets": offsets,
            "geometry": geometry,
            "gain": gain,
            "budget_scale": budget_scale,
        }

    def compute_components(self, deep: torch.Tensor, lateral: torch.Tensor):
        core_parts = super().compute_components(deep, lateral)
        core_output = core_parts["base"] + core_parts["correction"]
        precision = self.compute_precision_components(deep, lateral, core_output, core_parts)
        output = core_output + precision["correction"]

        return {
            **core_parts,
            "core_output": core_output,
            "precision_correction": precision["correction"],
            "precision_gate": precision["gate"],
            "precision_semantic": precision["semantic"],
            "precision_spatial_scale": precision["spatial_scale"],
            "precision_offsets": precision["offsets"],
            "precision_geometry": precision["geometry"],
            "precision_gain": precision["gain"],
            "precision_budget_scale": precision["budget_scale"],
            "output": output,
        }

    def forward(self, inputs):
        if not isinstance(inputs, (list, tuple)) or len(inputs) != 2:
            raise TypeError("UCRA2v4 expects [P4_deep, P3_lateral]")
        parts = self.compute_components(inputs[0], inputs[1])
        output = parts["output"]
        if self.record_diagnostics:
            self.latest_diagnostics = {
                key: value.detach() if torch.is_tensor(value) else value
                for key, value in parts.items()
                if key not in {"base", "core_output", "output"}
            }
        return output
