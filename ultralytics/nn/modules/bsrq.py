"""Same-index, identity-start BSRQ-YOLOv13 modules built on the verified B1 UDQ parent."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .block import FullPAD_Tunnel
from .conv import Conv


class BoundedSpectralConditioner(nn.Module):
    """Apply bounded, zero-sum RGB log gains while remaining an exact identity at initialization."""

    def __init__(self, max_log_gain: float = 0.08, hidden: int = 8, eps: float = 1e-6):
        super().__init__()
        self.max_log_gain, self.eps = float(max_log_gain), float(eps)
        hidden = int(hidden)
        if not 0.0 < self.max_log_gain <= 0.5:
            raise ValueError(f"max_log_gain must be in (0, 0.5], got {self.max_log_gain}")
        if hidden < 3 or self.eps <= 0.0:
            raise ValueError("hidden must be >= 3 and eps must be positive")
        self.mlp = nn.Sequential(nn.Linear(6, hidden), nn.SiLU(inplace=True), nn.Linear(hidden, 3))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def _log_gain(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        mean = xf.mean(dim=(2, 3)).clamp_min(self.eps)
        std = xf.var(dim=(2, 3), unbiased=False).add(self.eps).sqrt()
        log_mean = mean.log()
        chroma = log_mean - log_mean.mean(dim=1, keepdim=True)
        cues = torch.cat((chroma, torch.log1p((std / mean).clamp(0.0, 10.0))), dim=1)
        z = torch.tanh(self.mlp(cues))
        z = z - z.mean(dim=1, keepdim=True)
        z = z / z.abs().amax(dim=1, keepdim=True).clamp_min(1.0)
        return self.max_log_gain * z

    def forward(self, x: torch.Tensor, return_gain: bool = False):
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(f"BoundedSpectralConditioner expects RGB NCHW input, got {tuple(x.shape)}")
        gain = self._log_gain(x).exp().view(x.shape[0], 3, 1, 1).to(dtype=x.dtype)
        output = x * gain
        return (output, gain) if return_gain else output


class BoundedSpectralStem(Conv):
    """Conv-compatible layer-0 replacement that preserves pretrained ``conv.*`` and ``bn.*`` keys."""

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 3,
        s: int = 2,
        max_log_gain: float = 0.08,
        hidden: int = 8,
        p=None,
        g: int = 1,
        d: int = 1,
        act=True,
        eps: float = 1e-6,
    ):
        if int(c1) != 3:
            raise ValueError(f"BoundedSpectralStem requires RGB input, got c1={c1}")
        # Create inherited Conv first so all parent initialization and keys exactly match B1.
        super().__init__(c1, c2, k, s, p, g, d, act)
        # Extra parameters may not perturb downstream deterministic initialization.
        with torch.random.fork_rng(devices=[], enabled=True):
            self.spectral = BoundedSpectralConditioner(max_log_gain=max_log_gain, hidden=hidden, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(self.spectral(x))

    def forward_fuse(self, x: torch.Tensor) -> torch.Tensor:
        # BaseModel.fuse() dispatches every Conv subclass here after fusing its BN.
        return self.act(self.conv(self.spectral(x)))


class ReliabilityAdaptiveFullPAD(FullPAD_Tunnel):
    """Identity-start, bounded reliability modulation of the final P3 FullPAD enhanced path."""

    def __init__(self, max_modulation: float = 0.20, hidden: int = 8, detach_context: bool = True, eps: float = 1e-6):
        super().__init__()
        self.max_modulation, self.detach_context, self.eps = float(max_modulation), bool(detach_context), float(eps)
        hidden = int(hidden)
        if not 0.0 < self.max_modulation < 1.0:
            raise ValueError(f"max_modulation must be in (0, 1), got {self.max_modulation}")
        if hidden < 2 or self.eps <= 0.0:
            raise ValueError("hidden must be >= 2 and eps must be positive")
        with torch.random.fork_rng(devices=[], enabled=True):
            self.modulator = nn.Sequential(nn.Conv2d(3, hidden, 1), nn.SiLU(inplace=True), nn.Conv2d(hidden, 1, 1))
        nn.init.zeros_(self.modulator[-1].weight)
        nn.init.zeros_(self.modulator[-1].bias)

    def _cues(self, original: torch.Tensor, enhanced: torch.Tensor) -> torch.Tensor:
        original_context = original.detach() if self.detach_context else original
        enhanced_context = enhanced.detach() if self.detach_context else enhanced
        original_float, enhanced_float = original_context.float(), enhanced_context.float()
        dot = (original_float * enhanced_float).sum(dim=1, keepdim=True)
        original_norm = original_float.square().sum(dim=1, keepdim=True).add(self.eps).sqrt()
        enhanced_norm = enhanced_float.square().sum(dim=1, keepdim=True).add(self.eps).sqrt()
        agreement = ((dot / (original_norm * enhanced_norm + self.eps)).clamp(-1.0, 1.0) + 1.0) * 0.5
        original_energy = original_float.abs().mean(dim=1, keepdim=True)
        enhanced_energy = enhanced_float.abs().mean(dim=1, keepdim=True)
        log_ratio = torch.log((enhanced_energy + self.eps) / (original_energy + self.eps)).clamp(-4.0, 4.0) / 4.0
        local_energy = F.avg_pool2d(F.pad(original_energy, (1, 1, 1, 1), mode="replicate"), kernel_size=3, stride=1)
        detail_support = 1.0 - torch.exp(-((original_energy - local_energy).abs() / (local_energy.abs() + self.eps)).clamp(0.0, 8.0))
        return torch.cat((agreement, log_ratio, detail_support), dim=1).to(dtype=original.dtype)

    def forward(self, x) -> torch.Tensor:
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise ValueError("ReliabilityAdaptiveFullPAD expects [original, enhanced]")
        original, enhanced = x
        if original.shape != enhanced.shape or original.device != enhanced.device or original.dtype != enhanced.dtype:
            raise ValueError("FullPAD inputs must have matching shape, device, and dtype")
        factor = 1.0 + self.max_modulation * torch.tanh(self.modulator(self._cues(original, enhanced)))
        return original + self.gate.to(dtype=original.dtype) * factor * enhanced


class StatConsensusGate(nn.Module):
    """Per-scale bounded reliability factors built only from detached DFL statistics."""

    def __init__(self, nl: int = 3, max_deltas=(0.25, 0.15, 0.08), hidden: int = 8):
        super().__init__()
        self.nl = int(nl)
        hidden = int(hidden)
        if self.nl <= 0 or len(max_deltas) != self.nl or hidden < 2:
            raise ValueError("invalid nl, max_deltas, or hidden")
        deltas = torch.tensor(max_deltas, dtype=torch.float32)
        if torch.any((deltas <= 0.0) | (deltas >= 1.0)):
            raise ValueError("each max_delta must be in (0, 1)")
        self.register_buffer("max_deltas", deltas, persistent=True)
        with torch.random.fork_rng(devices=[], enabled=True):
            self.gates = nn.ModuleList(
                nn.Sequential(nn.Conv2d(3, hidden, 1), nn.SiLU(inplace=True), nn.Conv2d(hidden, 1, 1)) for _ in range(self.nl)
            )
        for gate in self.gates:
            nn.init.zeros_(gate[-1].weight)
            nn.init.zeros_(gate[-1].bias)

    def forward(self, statistics: torch.Tensor, level_index: int) -> torch.Tensor:
        if statistics.ndim != 4 or statistics.shape[1] != 12:
            raise ValueError(f"statistics must be BCHW with C=12, got {tuple(statistics.shape)}")
        index = int(level_index)
        if not 0 <= index < self.nl:
            raise IndexError(f"invalid level_index={index}")
        detached = statistics.detach().float()
        cues = torch.cat(
            (detached.mean(dim=1, keepdim=True), detached.std(dim=1, keepdim=True, unbiased=False), detached.amin(dim=1, keepdim=True)), dim=1
        ).to(dtype=statistics.dtype)
        bound = self.max_deltas[index].to(device=statistics.device, dtype=statistics.dtype)
        return 1.0 + bound * torch.tanh(self.gates[index](cues))


class SCQQualityFusion(nn.Module):
    """Identity-start replacement for B1's quality-statistic fusion only."""

    def __init__(self, stat_strengths=(1.0, 0.5, 0.25), max_deltas=(0.25, 0.15, 0.08), hidden: int = 8):
        super().__init__()
        if len(stat_strengths) != len(max_deltas) or any(float(value) < 0.0 for value in stat_strengths):
            raise ValueError("stat_strengths and max_deltas must have equal valid lengths")
        self.register_buffer("stat_strengths", torch.tensor(stat_strengths, dtype=torch.float32), persistent=True)
        self.reliability = StatConsensusGate(nl=len(stat_strengths), max_deltas=max_deltas, hidden=hidden)

    def forward(self, quality_feature: torch.Tensor, quality_statistic: torch.Tensor, statistics: torch.Tensor, level_index: int) -> torch.Tensor:
        if quality_feature.shape != quality_statistic.shape or quality_feature.ndim != 4 or quality_feature.shape[1] != 1:
            raise ValueError("quality feature/statistic logits must share shape [B, 1, H, W]")
        if statistics.shape[0] != quality_feature.shape[0] or statistics.shape[-2:] != quality_feature.shape[-2:]:
            raise ValueError("statistics and quality logits are spatially incompatible")
        index = int(level_index)
        if not 0 <= index < len(self.stat_strengths):
            raise IndexError(f"invalid level_index={index}")
        strength = self.stat_strengths[index].to(device=quality_feature.device, dtype=quality_feature.dtype)
        return quality_feature + strength * self.reliability(statistics, index) * quality_statistic
