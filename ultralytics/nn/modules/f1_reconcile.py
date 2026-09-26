from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .head import Detect

__all__ = ("F1ReconcileDetect", "F1ReconcileAdapter")


class _ConvGNAct(nn.Module):
    """Small-batch-friendly projection for the auxiliary classification adapter."""

    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, groups: int = 1, gn_groups: int = 8):
        super().__init__()
        c1, c2, k, s, groups = map(int, (c1, c2, k, s, groups))
        if min(c1, c2, k, s, groups) <= 0:
            raise ValueError("invalid convolution arguments")
        if c1 % groups != 0 or c2 % groups != 0:
            raise ValueError(f"groups={groups} must divide c1={c1} and c2={c2}")
        self.conv = nn.Conv2d(c1, c2, k, s, k // 2, groups=groups, bias=False)
        ng = min(int(gn_groups), c2)
        while ng > 1 and c2 % ng:
            ng -= 1
        self.norm = nn.GroupNorm(ng, c2)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class F1ReconcileAdapter(nn.Module):
    """
    Gradient-isolated P3 classification-logit reconciler.

    The deployed Detect regression path remains untouched. The adapter uses detached
    P3/P4 context plus detached P3 DFL confidence to generate a bounded residual
    only for the P3 classification logits.

    Exact-init invariant:
        gain_raw == 0  ->  output_logits == base_logits bit-for-bit.
    """

    def __init__(
        self,
        c_p3: int,
        c_p4: int,
        nc: int,
        reg_max: int,
        max_delta: float = 0.75,
        loc_floor: float = 0.50,
        reduction: int = 4,
        detach_context: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.c_p3 = int(c_p3)
        self.c_p4 = int(c_p4)
        self.nc = int(nc)
        self.reg_max = int(reg_max)
        self.max_delta = float(max_delta)
        self.loc_floor = float(loc_floor)
        self.detach_context = bool(detach_context)
        self.eps = float(eps)

        if min(self.c_p3, self.c_p4, self.nc) <= 0:
            raise ValueError("c_p3, c_p4 and nc must be positive")
        if self.reg_max <= 1:
            raise ValueError("reg_max must be > 1")
        if not 0.0 <= self.max_delta <= 2.0:
            raise ValueError("max_delta must be in [0, 2]")
        if not 0.0 <= self.loc_floor <= 1.0:
            raise ValueError("loc_floor must be in [0, 1]")
        if int(reduction) < 1:
            raise ValueError("reduction must be >= 1")
        if self.eps <= 0.0:
            raise ValueError("eps must be positive")

        hidden = max(16, min(64, min(self.c_p3, self.c_p4) // int(reduction)))
        self.hidden = hidden

        blur = torch.tensor(
            ((1.0, 2.0, 1.0), (2.0, 4.0, 2.0), (1.0, 2.0, 1.0)),
            dtype=torch.float32,
        )
        self.register_buffer("blur_kernel", (blur / blur.sum())[None, None], persistent=False)

        self.p3_proj = _ConvGNAct(self.c_p3, hidden, 1, 1)
        self.p4_proj = _ConvGNAct(self.c_p4, hidden, 1, 1)

        self.fuse = nn.Sequential(
            _ConvGNAct(3 * hidden, 3 * hidden, 3, 1, groups=3 * hidden),
            _ConvGNAct(3 * hidden, hidden, 1, 1),
        )
        self.aux_logits = nn.Conv2d(hidden, self.nc, 1, bias=True)

        # Small non-zero candidate logits + exactly-zero global gain:
        # the first optimizer step trains gain_raw without perturbing the base head;
        # subsequent steps unlock the adapter internals.
        nn.init.normal_(self.aux_logits.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.aux_logits.bias)
        self.gain_raw = nn.Parameter(torch.zeros(1, dtype=torch.float32))

        self.record_diagnostics = False
        self.latest_diagnostics = {}

    def set_diagnostics(self, enabled: bool = True):
        self.record_diagnostics = bool(enabled)
        if not self.record_diagnostics:
            self.latest_diagnostics = {}
        return self

    def _blur(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.blur_kernel.to(device=x.device, dtype=x.dtype).repeat(x.shape[1], 1, 1, 1)
        return F.conv2d(F.pad(x, (1, 1, 1, 1), mode="replicate"), weight, groups=x.shape[1])

    def _semantic_support(self, p3_embed: torch.Tensor, p4_embed: torch.Tensor):
        p3_low = self._blur(p3_embed)
        p4_low = self._blur(p4_embed)

        p3_norm = F.normalize(p3_low.float(), dim=1, eps=self.eps)
        p4_norm = F.normalize(p4_low.float(), dim=1, eps=self.eps)
        agreement = ((p3_norm * p4_norm).sum(dim=1, keepdim=True) + 1.0).mul(0.5).clamp(0.0, 1.0)

        p3_energy = p3_low.float().abs().mean(dim=1, keepdim=True)
        p4_energy = p4_low.float().abs().mean(dim=1, keepdim=True)
        discrepancy = (p3_low.float() - p4_low.float()).abs().mean(dim=1, keepdim=True)
        consistency = torch.exp(
            -(discrepancy / (p3_energy + p4_energy + self.eps)).clamp(0.0, 8.0)
        )

        # No epsilon inside the product before sqrt: if both supports are exactly
        # zero, semantic support must remain exactly zero.
        semantic = torch.sqrt((agreement * consistency).clamp_min(0.0)).clamp(0.0, 1.0)
        return semantic, agreement, consistency

    def _localization_guard(self, box_logits: torch.Tensor) -> torch.Tensor:
        if box_logits.ndim != 4:
            raise ValueError("box_logits must be NCHW")
        b, channels, h, w = box_logits.shape
        expected = 4 * self.reg_max
        if channels != expected:
            raise ValueError(f"expected {expected} DFL channels, got {channels}")

        probability = (
            box_logits.detach()
            .float()
            .view(b, 4, self.reg_max, h, w)
            .softmax(dim=2)
        )
        entropy = -(probability.clamp_min(self.eps).log() * probability).sum(dim=2)
        entropy = entropy / math.log(self.reg_max)
        confidence = (1.0 - entropy.mean(dim=1, keepdim=True)).clamp(0.0, 1.0)
        return self.loc_floor + (1.0 - self.loc_floor) * confidence

    @staticmethod
    def _miss_ambiguity(base_logits: torch.Tensor) -> torch.Tensor:
        """
        Focus the adapter on uncertain/under-confident locations.

        g(p)=4p(1-p)*(1-p)^0.5:
        - near-zero background scores receive almost no correction;
        - mid/low confidence candidates receive the most correction;
        - already-high-confidence detections are changed conservatively.
        """
        p = base_logits.detach().float().sigmoid()
        return (4.0 * p * (1.0 - p) * torch.sqrt((1.0 - p).clamp_min(0.0))).clamp(0.0, 1.0)

    def compute_components(
        self,
        p3: torch.Tensor,
        p4: torch.Tensor,
        base_logits: torch.Tensor,
        box_logits: torch.Tensor,
    ):
        if p3.ndim != 4 or p4.ndim != 4 or base_logits.ndim != 4 or box_logits.ndim != 4:
            raise ValueError("F1ReconcileAdapter expects NCHW tensors")
        if p3.shape[0] != p4.shape[0] or p3.shape[0] != base_logits.shape[0]:
            raise ValueError("batch sizes differ")
        if p3.shape[1] != self.c_p3 or p4.shape[1] != self.c_p4:
            raise ValueError(
                f"expected P3/P4 channels {self.c_p3}/{self.c_p4}, got {p3.shape[1]}/{p4.shape[1]}"
            )
        if base_logits.shape[1] != self.nc:
            raise ValueError(f"expected {self.nc} class logits, got {base_logits.shape[1]}")
        if base_logits.shape[-2:] != p3.shape[-2:] or box_logits.shape[-2:] != p3.shape[-2:]:
            raise ValueError("P3, base_logits and box_logits must share spatial size")
        if p3.device != p4.device or p3.dtype != p4.dtype:
            raise ValueError("P3/P4 must share device and dtype")

        p3_source = p3.detach() if self.detach_context else p3
        p4_source = p4.detach() if self.detach_context else p4

        p3_embed = self.p3_proj(p3_source)
        p4_embed = F.interpolate(
            self.p4_proj(p4_source),
            size=p3.shape[-2:],
            mode="nearest",
        )

        semantic, agreement, consistency = self._semantic_support(p3_embed, p4_embed)
        fused = self.fuse(torch.cat((p3_embed, p4_embed, (p3_embed - p4_embed).abs()), dim=1))
        candidate_logits = self.aux_logits(fused)

        ambiguity = self._miss_ambiguity(base_logits)
        loc_guard = self._localization_guard(box_logits)
        support = semantic.detach().to(base_logits.dtype) * loc_guard.detach().to(base_logits.dtype) * ambiguity.to(base_logits.dtype)

        gain = self.max_delta * torch.tanh(self.gain_raw)
        delta_logits = gain.to(base_logits.dtype) * torch.tanh(candidate_logits) * support
        output_logits = base_logits + delta_logits

        return {
            "output_logits": output_logits,
            "delta_logits": delta_logits,
            "gain": gain,
            "semantic": semantic,
            "agreement": agreement,
            "consistency": consistency,
            "ambiguity": ambiguity,
            "loc_guard": loc_guard,
            "support": support,
            "candidate_logits": candidate_logits,
        }

    def forward(
        self,
        p3: torch.Tensor,
        p4: torch.Tensor,
        base_logits: torch.Tensor,
        box_logits: torch.Tensor,
    ) -> torch.Tensor:
        parts = self.compute_components(p3, p4, base_logits, box_logits)
        if self.record_diagnostics:
            self.latest_diagnostics = {
                key: value.detach() if torch.is_tensor(value) else value
                for key, value in parts.items()
                if key != "output_logits"
            }
        return parts["output_logits"]


class F1ReconcileDetect(Detect):
    """
    Three-scale Detect with a bounded P3 classification-only reconciliation branch.

    Intended use:
        A4 UCRA-v2 neck + F1ReconcileDetect

    Deployed box regression remains exactly the inherited Detect cv2 towers.
    P4/P5 classification remains exactly the inherited Detect cv3 towers.
    Only P3 classification logits may receive the bounded residual.
    """

    def __init__(
        self,
        nc: int = 80,
        max_delta: float = 0.75,
        loc_floor: float = 0.50,
        reduction: int = 4,
        detach_context: bool = True,
        eps: float = 1e-6,
        ch=(),
    ):
        if not isinstance(ch, (list, tuple)) or len(ch) != 3:
            raise ValueError(f"F1ReconcileDetect requires P3/P4/P5 channels, got {ch}")

        super().__init__(nc=nc, ch=ch)
        if self.end2end:
            raise NotImplementedError("F1ReconcileDetect supports only the standard one-to-many path")
        if self.nl != 3:
            raise ValueError(f"F1ReconcileDetect requires exactly three detection levels, got {self.nl}")

        # Detect is the last layer; fork_rng still matters for exact same-seed
        # initialization/equivalence tests and reproducible ablations.
        with torch.random.fork_rng(devices=[], enabled=True):
            self.f1_adapter = F1ReconcileAdapter(
                c_p3=int(ch[0]),
                c_p4=int(ch[1]),
                nc=int(nc),
                reg_max=int(self.reg_max),
                max_delta=max_delta,
                loc_floor=loc_floor,
                reduction=reduction,
                detach_context=detach_context,
                eps=eps,
            )

        self.latest_diagnostics = None

    def set_diagnostics(self, enabled: bool = True):
        self.f1_adapter.set_diagnostics(enabled)
        return self

    def forward(self, x):
        if not isinstance(x, (list, tuple)) or len(x) != 3:
            raise ValueError("F1ReconcileDetect expects [P3, P4, P5]")

        p3, p4, p5 = x
        features = (p3, p4, p5)
        outputs = []

        for index, feature in enumerate(features):
            box_logits = self.cv2[index](feature)
            base_cls_logits = self.cv3[index](feature)

            if index == 0:
                cls_logits = self.f1_adapter(
                    p3=p3,
                    p4=p4,
                    base_logits=base_cls_logits,
                    box_logits=box_logits,
                )
                if self.training and self.f1_adapter.record_diagnostics:
                    self.latest_diagnostics = self.f1_adapter.latest_diagnostics
            else:
                cls_logits = base_cls_logits

            outputs.append(torch.cat((box_logits, cls_logits), dim=1))

        if self.training:
            return outputs

        prediction = self._inference(outputs)
        return prediction if self.export else (prediction, outputs)
