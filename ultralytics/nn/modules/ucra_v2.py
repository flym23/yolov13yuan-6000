from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ("UCRA1v2", "UCRA2v2")


class _ConvGNAct(nn.Module):
    """Small-batch-friendly projection used only inside the new residual branch."""

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 1,
        s: int = 1,
        groups: int = 1,
        act: bool = True,
        gn_groups: int = 8,
    ):
        super().__init__()
        if c1 <= 0 or c2 <= 0 or k <= 0 or s <= 0 or groups <= 0:
            raise ValueError("invalid convolution arguments")
        if c1 % groups != 0 or c2 % groups != 0:
            raise ValueError(f"groups={groups} must divide c1={c1} and c2={c2}")

        self.conv = nn.Conv2d(c1, c2, k, s, k // 2, groups=groups, bias=False)

        ng = min(int(gn_groups), c2)
        while ng > 1 and c2 % ng:
            ng -= 1
        self.norm = nn.GroupNorm(ng, c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class _ResidualBudget(nn.Module):
    """Strict RMS limiter for the learned correction."""

    def __init__(
        self,
        max_ratio: float,
        per_channel: bool,
        eps: float = 1e-6,
        detach_scale: bool = True,
    ):
        super().__init__()
        self.max_ratio = float(max_ratio)
        self.per_channel = bool(per_channel)
        self.eps = float(eps)
        self.detach_scale = bool(detach_scale)

        if not 0.0 < self.max_ratio <= 1.0:
            raise ValueError("max_ratio must be in (0, 1]")
        if self.eps <= 0.0:
            raise ValueError("eps must be positive")

    def forward(self, base: torch.Tensor, correction: torch.Tensor):
        if base.shape != correction.shape:
            raise ValueError("base and correction must share shape")

        dims = (2, 3) if self.per_channel else (1, 2, 3)

        # Do not add eps to base RMS: zero-energy main-path channels/samples must
        # receive exactly zero correction.
        base_rms = base.float().square().mean(dim=dims, keepdim=True).sqrt()
        corr_rms = (
            correction.float().square().mean(dim=dims, keepdim=True).add(self.eps).sqrt()
        )

        scale = torch.minimum(
            torch.ones_like(corr_rms),
            self.max_ratio * base_rms / corr_rms.clamp_min(self.eps),
        )
        if self.detach_scale:
            scale = scale.detach()

        return (correction.float() * scale).to(dtype=base.dtype), scale


class _UCRAv2Up(nn.Module):
    """
    UCRA-v2 common implementation.

    Design invariants
    -----------------
    1. Main path is immutable nearest-neighbour upsampling.
    2. All new behavior is a zero-start residual branch.
    3. Lateral features guide alignment but do not replace the original YOLOv13
       Concat + DSC3k2 path.
    4. Learned offsets are bounded in source-feature pixels.
    5. Residual energy is explicitly bounded.
    """

    def __init__(
        self,
        c_deep: int,
        c_lateral: int,
        mode: str,
        scale: int = 2,
        reduction: int = 4,
        sample_groups: int = 4,
        max_offset: float = 0.25,
        max_residual_ratio: float = 0.08,
        reliability_floor: float = 0.20,
        strict_scale: bool = True,
        detach_reliability: bool = True,
        detach_budget: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()

        self.c_deep = int(c_deep)
        self.c_lateral = int(c_lateral)
        self.mode = str(mode)
        self.scale = int(scale)
        self.max_offset = float(max_offset)
        self.reliability_floor = float(reliability_floor)
        self.strict_scale = bool(strict_scale)
        self.detach_reliability = bool(detach_reliability)
        self.eps = float(eps)

        if self.c_deep <= 0 or self.c_lateral <= 0:
            raise ValueError("channel counts must be positive")
        if self.mode not in {"semantic", "detail"}:
            raise ValueError("mode must be 'semantic' or 'detail'")
        if self.scale <= 1:
            raise ValueError("scale must be > 1")
        if int(reduction) < 1:
            raise ValueError("reduction must be >= 1")
        if int(sample_groups) < 1:
            raise ValueError("sample_groups must be >= 1")
        if not 0.0 <= self.max_offset <= 1.0:
            raise ValueError("max_offset must be in [0, 1]")
        if not 0.0 <= self.reliability_floor <= 1.0:
            raise ValueError("reliability_floor must be in [0, 1]")
        if self.eps <= 0.0:
            raise ValueError("eps must be positive")

        self.hidden = max(
            16,
            min(64, min(self.c_deep, self.c_lateral) // int(reduction)),
        )
        self.sample_groups = max(1, math.gcd(self.c_deep, int(sample_groups)))
        self.group_channels = self.c_deep // self.sample_groups

        blur = torch.tensor(
            ((1.0, 2.0, 1.0), (2.0, 4.0, 2.0), (1.0, 2.0, 1.0)),
            dtype=torch.float32,
        )
        self.register_buffer(
            "blur_kernel",
            (blur / blur.sum())[None, None],
            persistent=False,
        )

        sobel_x = torch.tensor(
            ((-1.0, 0.0, 1.0), (-2.0, 0.0, 2.0), (-1.0, 0.0, 1.0)),
            dtype=torch.float32,
        ) / 8.0
        self.register_buffer("sobel_x", sobel_x[None, None], persistent=False)
        self.register_buffer(
            "sobel_y",
            sobel_x.t().contiguous()[None, None],
            persistent=False,
        )

        # nn.Upsample itself owns no random parameters. The fork prevents these
        # new parameters from changing the RNG stream used by downstream
        # pretrained-compatible YOLOv13 layers when training from scratch.
        with torch.random.fork_rng(devices=[], enabled=True):
            self.deep_proj = _ConvGNAct(self.c_deep, self.hidden, 1, 1)
            self.lateral_proj = _ConvGNAct(self.c_lateral, self.hidden, 1, 1)

            # Lightweight DySample-style continuous offset predictor:
            # depthwise spatial mixing -> pointwise compression -> offsets.
            self.offset_head = nn.Sequential(
                _ConvGNAct(
                    self.hidden * 3,
                    self.hidden * 3,
                    3,
                    1,
                    groups=self.hidden * 3,
                ),
                _ConvGNAct(self.hidden * 3, self.hidden, 1, 1),
                nn.Conv2d(
                    self.hidden,
                    2 * self.sample_groups,
                    1,
                    bias=True,
                ),
            )
            nn.init.zeros_(self.offset_head[-1].weight)
            nn.init.zeros_(self.offset_head[-1].bias)

            self.context_out = nn.Conv2d(
                self.hidden,
                self.c_deep,
                1,
                bias=False,
            )

            residual_groups = max(1, math.gcd(self.c_deep, 4))
            self.residual_out = nn.Conv2d(
                self.c_deep,
                self.c_deep,
                1,
                groups=residual_groups,
                bias=False,
            )

            # Exact baseline at initialization.
            nn.init.zeros_(self.residual_out.weight)

        self.budget = _ResidualBudget(
            max_ratio=max_residual_ratio,
            per_channel=(self.mode == "semantic"),
            eps=self.eps,
            detach_scale=detach_budget,
        )

        self.record_diagnostics = False
        self.latest_diagnostics = {}

    def set_diagnostics(self, enabled: bool = True):
        self.record_diagnostics = bool(enabled)
        if not self.record_diagnostics:
            self.latest_diagnostics = {}
        return self

    def _validate_inputs(
        self,
        deep: torch.Tensor,
        lateral: torch.Tensor,
    ):
        if deep.ndim != 4 or lateral.ndim != 4:
            raise ValueError("expected NCHW tensors")
        if deep.shape[0] != lateral.shape[0]:
            raise ValueError("deep/lateral batch sizes differ")
        if deep.shape[1] != self.c_deep or lateral.shape[1] != self.c_lateral:
            raise ValueError(
                f"expected channels {self.c_deep}/{self.c_lateral}, "
                f"got {deep.shape[1]}/{lateral.shape[1]}"
            )
        if deep.device != lateral.device or deep.dtype != lateral.dtype:
            raise ValueError("deep/lateral must share device and dtype")

        expected = (
            deep.shape[-2] * self.scale,
            deep.shape[-1] * self.scale,
        )
        if self.strict_scale and tuple(lateral.shape[-2:]) != expected:
            raise ValueError(
                f"expected lateral size {expected}, "
                f"got {tuple(lateral.shape[-2:])}"
            )

    def _blur(self, x: torch.Tensor) -> torch.Tensor:
        channels = x.shape[1]
        kernel = (
            self.blur_kernel.to(device=x.device, dtype=x.dtype)
            .repeat(channels, 1, 1, 1)
        )
        return F.conv2d(
            F.pad(x, (1, 1, 1, 1), mode="replicate"),
            kernel,
            groups=channels,
        )

    def _reliability(
        self,
        deep_embed: torch.Tensor,
        lateral_embed: torch.Tensor,
    ):
        deep_low = self._blur(deep_embed)
        lateral_low = self._blur(lateral_embed)
        lateral_high = lateral_embed - lateral_low

        # Low-frequency cross-scale semantic agreement.
        deep_norm = F.normalize(
            deep_low.float(),
            dim=1,
            eps=self.eps,
        )
        lateral_norm = F.normalize(
            lateral_low.float(),
            dim=1,
            eps=self.eps,
        )
        agreement = (
            (deep_norm * lateral_norm)
            .sum(dim=1, keepdim=True)
            .add(1.0)
            .mul(0.5)
            .clamp(0.0, 1.0)
        )

        deep_energy = deep_low.float().abs().mean(
            dim=1,
            keepdim=True,
        )
        lateral_energy = lateral_low.float().abs().mean(
            dim=1,
            keepdim=True,
        )

        discrepancy = (
            deep_low.float() - lateral_low.float()
        ).abs().mean(dim=1, keepdim=True)

        consistency = torch.exp(
            -(
                discrepancy
                / (deep_energy + lateral_energy + self.eps)
            ).clamp(0.0, 8.0)
        )

        semantic_reliability = (
            0.5 * agreement + 0.5 * consistency
        ).clamp(0.0, 1.0)

        # Detail reliability:
        # edge saliency × structure-tensor orientation coherence.
        # This discriminates coherent object boundaries from isotropic
        # high-frequency underwater noise better than raw high-pass energy.
        summary = (
            lateral_embed.float()
            .square()
            .mean(dim=1, keepdim=True)
            .add(self.eps)
            .sqrt()
        )

        padded = F.pad(
            summary,
            (1, 1, 1, 1),
            mode="replicate",
        )
        gx = F.conv2d(
            padded,
            self.sobel_x.to(
                device=summary.device,
                dtype=summary.dtype,
            ),
        )
        gy = F.conv2d(
            padded,
            self.sobel_y.to(
                device=summary.device,
                dtype=summary.dtype,
            ),
        )
        grad_mag = (
            gx.square() + gy.square() + self.eps
        ).sqrt()

        grad_mean = grad_mag.mean(
            dim=(2, 3),
            keepdim=True,
        )
        grad_std = (
            grad_mag.var(
                dim=(2, 3),
                keepdim=True,
                unbiased=False,
            )
            .add(self.eps)
            .sqrt()
        )

        relative_edge = torch.sigmoid(
            (grad_mag - grad_mean) / grad_std
        )
        absolute_edge = 1.0 - torch.exp(
            -(
                grad_mag
                / (lateral_energy + self.eps)
            ).clamp(0.0, 8.0)
        )
        edge_strength = (
            absolute_edge * relative_edge
        ).clamp(0.0, 1.0).sqrt()

        jxx = F.avg_pool2d(
            F.pad(
                gx.square(),
                (1, 1, 1, 1),
                mode="replicate",
            ),
            3,
            1,
        )
        jyy = F.avg_pool2d(
            F.pad(
                gy.square(),
                (1, 1, 1, 1),
                mode="replicate",
            ),
            3,
            1,
        )
        jxy = F.avg_pool2d(
            F.pad(
                gx * gy,
                (1, 1, 1, 1),
                mode="replicate",
            ),
            3,
            1,
        )

        coherence = (
            (
                (jxx - jyy).square()
                + 4.0 * jxy.square()
                + self.eps
            ).sqrt()
            / (jxx + jyy + self.eps)
        ).clamp(0.0, 1.0)

        detail_reliability = (
            edge_strength
            * (0.25 + 0.75 * coherence)
        ).clamp(0.0, 1.0)

        if self.mode == "semantic":
            reliability = semantic_reliability
        else:
            # P4->P3 gives more weight to coherent detail, while preserving
            # a semantic-consistency term to avoid indiscriminate noise release.
            reliability = (
                0.35 * semantic_reliability
                + 0.65 * detail_reliability
            ).clamp(0.0, 1.0)

        permission = (
            self.reliability_floor
            + (1.0 - self.reliability_floor)
            * reliability
        )

        if self.detach_reliability:
            permission = permission.detach()
            agreement = agreement.detach()
            consistency = consistency.detach()
            detail_reliability = detail_reliability.detach()

        return (
            permission.to(dtype=deep_embed.dtype),
            agreement,
            consistency,
            detail_reliability,
            deep_low,
            lateral_low,
            lateral_high,
        )

    @staticmethod
    def _base_grid(
        batch: int,
        out_h: int,
        out_w: int,
        device,
        dtype,
    ):
        # align_corners=False sampling grid.
        yy = (
            2.0
            * (
                torch.arange(
                    out_h,
                    device=device,
                    dtype=torch.float32,
                )
                + 0.5
            )
            / out_h
            - 1.0
        )
        xx = (
            2.0
            * (
                torch.arange(
                    out_w,
                    device=device,
                    dtype=torch.float32,
                )
                + 0.5
            )
            / out_w
            - 1.0
        )
        gy, gx = torch.meshgrid(
            yy,
            xx,
            indexing="ij",
        )
        grid = (
            torch.stack((gx, gy), dim=-1)
            .unsqueeze(0)
            .expand(batch, -1, -1, -1)
        )
        return grid.to(dtype=dtype)

    def _dynamic_sample(
        self,
        deep: torch.Tensor,
        offsets: torch.Tensor,
        out_hw: tuple[int, int],
    ):
        batch, channels, in_h, in_w = deep.shape
        out_h, out_w = out_hw
        groups = self.sample_groups
        group_channels = self.group_channels

        expected = (
            batch,
            2 * groups,
            out_h,
            out_w,
        )
        if tuple(offsets.shape) != expected:
            raise ValueError(
                f"unexpected offset shape "
                f"{tuple(offsets.shape)}, expected {expected}"
            )

        # Offset unit = one source-feature pixel.
        offset_view = offsets.view(
            batch,
            groups,
            2,
            out_h,
            out_w,
        ).float()

        off_x = (
            offset_view[:, :, 0]
            * (2.0 / max(in_w, 1))
        )
        off_y = (
            offset_view[:, :, 1]
            * (2.0 / max(in_h, 1))
        )
        off_grid = torch.stack(
            (off_x, off_y),
            dim=-1,
        )

        base_grid = self._base_grid(
            batch,
            out_h,
            out_w,
            deep.device,
            torch.float32,
        ).unsqueeze(1)

        grid = (
            base_grid + off_grid
        ).reshape(
            batch * groups,
            out_h,
            out_w,
            2,
        ).to(dtype=deep.dtype)

        deep_grouped = deep.reshape(
            batch,
            groups,
            group_channels,
            in_h,
            in_w,
        ).reshape(
            batch * groups,
            group_channels,
            in_h,
            in_w,
        )

        sampled = F.grid_sample(
            deep_grouped,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )

        return sampled.reshape(
            batch,
            groups,
            group_channels,
            out_h,
            out_w,
        ).reshape(
            batch,
            channels,
            out_h,
            out_w,
        )

    @staticmethod
    def _local_mean(
        x: torch.Tensor,
    ) -> torch.Tensor:
        return F.avg_pool2d(
            F.pad(
                x,
                (1, 1, 1, 1),
                mode="replicate",
            ),
            3,
            1,
        )

    def _orthogonalize(
        self,
        base: torch.Tensor,
        correction: torch.Tensor,
    ) -> torch.Tensor:
        base_f = base.float()
        corr_f = correction.float()

        denominator = base_f.square().sum(
            dim=1,
            keepdim=True,
        )
        projection = (
            (base_f * corr_f).sum(
                dim=1,
                keepdim=True,
            )
            / (denominator + self.eps)
        )

        return (
            corr_f - projection * base_f
        ).to(dtype=correction.dtype)

    def compute_components(
        self,
        deep: torch.Tensor,
        lateral: torch.Tensor,
    ):
        self._validate_inputs(deep, lateral)

        target_hw = tuple(lateral.shape[-2:])

        # Immutable original-YOLO main path.
        base = F.interpolate(
            deep,
            size=target_hw,
            mode="nearest",
        )

        deep_embed = F.interpolate(
            self.deep_proj(deep),
            size=target_hw,
            mode="bilinear",
            align_corners=False,
        )
        lateral_embed = self.lateral_proj(
            lateral
        )

        (
            permission,
            agreement,
            consistency,
            detail_reliability,
            deep_low,
            lateral_low,
            lateral_high,
        ) = self._reliability(
            deep_embed,
            lateral_embed,
        )

        cue = torch.cat(
            (
                deep_embed,
                lateral_embed,
                (deep_embed - lateral_embed).abs(),
            ),
            dim=1,
        )

        raw_offsets = self.offset_head(cue)

        # Low-reliability positions are not allowed to make large geometric moves.
        offset_permission = permission.repeat(
            1,
            2 * self.sample_groups,
            1,
            1,
        )
        offsets = (
            self.max_offset
            * torch.tanh(raw_offsets)
            * offset_permission
        )

        sampled = self._dynamic_sample(
            deep,
            offsets,
            target_hw,
        )
        sample_delta = sampled - base

        if self.mode == "semantic":
            # P5->P4: restore low-frequency semantic discrepancy conservatively.
            context_hidden = (
                lateral_low - deep_low
            )
            sample_weight = 0.60
        else:
            # P4->P3: only coherent lateral high-frequency evidence is released.
            context_hidden = (
                lateral_high * permission
            )
            sample_weight = 0.40

        context_delta = self.context_out(
            context_hidden
        )

        candidate = permission * (
            sample_weight * sample_delta
            + (1.0 - sample_weight)
            * context_delta
        )

        raw_correction = self.residual_out(
            candidate
        )

        if self.mode == "detail":
            # For the small-target stage, forbid the auxiliary branch from
            # degenerating into hidden low-frequency gain scaling.
            raw_correction = (
                raw_correction
                - self._local_mean(
                    raw_correction.float()
                ).to(raw_correction.dtype)
            )
            raw_correction = self._orthogonalize(
                base,
                raw_correction,
            )

        correction, budget_scale = self.budget(
            base,
            raw_correction,
        )

        return {
            "base": base,
            "correction": correction,
            "permission": permission,
            "agreement": agreement,
            "consistency": consistency,
            "detail_reliability": detail_reliability,
            "offsets": offsets,
            "budget_scale": budget_scale,
        }

    def forward(self, inputs):
        if (
            not isinstance(inputs, (list, tuple))
            or len(inputs) != 2
        ):
            raise TypeError(
                f"{self.__class__.__name__} "
                "expects [deep, lateral]"
            )

        parts = self.compute_components(
            inputs[0],
            inputs[1],
        )
        output = (
            parts["base"]
            + parts["correction"]
        )

        if self.record_diagnostics:
            self.latest_diagnostics = {
                key: (
                    value.detach()
                    if torch.is_tensor(value)
                    else value
                )
                for key, value in parts.items()
                if key != "base"
            }

        return output


class UCRA1v2(_UCRAv2Up):
    """
    P5 -> P4 semantic stage.

    Default maximum offset = 0.25 source pixels.
    Default residual budget = 8% per-channel RMS.
    """

    def __init__(
        self,
        c_deep: int,
        c_lateral: int,
        scale: int = 2,
        reduction: int = 4,
        sample_groups: int = 4,
        max_offset: float = 0.25,
        max_residual_ratio: float = 0.08,
        reliability_floor: float = 0.25,
        strict_scale: bool = True,
        detach_reliability: bool = True,
        detach_budget: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__(
            c_deep=c_deep,
            c_lateral=c_lateral,
            mode="semantic",
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


class UCRA2v2(_UCRAv2Up):
    """
    P4 -> P3 detail/small-object stage.

    Default maximum offset = 0.50 source pixels.
    Default residual budget = 10% sample-wise RMS.
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
    ):
        super().__init__(
            c_deep=c_deep,
            c_lateral=c_lateral,
            mode="detail",
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

