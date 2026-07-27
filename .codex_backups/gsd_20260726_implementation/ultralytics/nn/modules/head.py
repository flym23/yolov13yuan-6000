# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Model head modules."""

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import constant_, xavier_uniform_

from ultralytics.utils.tal import TORCH_1_10, dist2bbox, dist2rbox, make_anchors

from .block import DFL, BNContrastiveHead, ContrastiveHead, Proto
from .conv import Conv, DWConv
from .transformer import MLP, DeformableTransformerDecoder, DeformableTransformerDecoderLayer
from .utils import bias_init_with_prob, linear_init

__all__ = (
    "Detect",
    "HRCTDetect",
    "NonUniformDFL",
    "SUDLDetect",
    "SBRHDetect",
    "P3TaskAdapter",
    "P3DecoupledDetect",
    "HighResolutionEvidenceEncoder",
    "AmbiguityReactivationGate",
    "P2RecallReactivation",
    "RAMPDetect",
    "GradientIsolatedShallowEncoder",
    "ClassPrototypeComplementaryRecovery",
    "CPCRDetect",
    "SemanticContextBridge",
    "BoundaryContextBridge",
    "CSTDDetect",
    "Segment",
    "Pose",
    "Classify",
    "OBB",
    "RTDETRDecoder",
    "v10Detect",
)


class Detect(nn.Module):
    """YOLO Detect head for detection models."""

    dynamic = False  # force grid reconstruction
    export = False  # export mode
    format = None  # export format
    end2end = False  # end2end
    max_det = 300  # max_det
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init
    legacy = False  # backward compatibility for v3/v5/v8/v9 models

    def __init__(self, nc=80, ch=()):
        """Initializes the YOLO detection layer with specified number of classes and channels."""
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
        )
        self.cv3 = (
            nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
            if self.legacy
            else nn.ModuleList(
                nn.Sequential(
                    nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                    nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                    nn.Conv2d(c3, self.nc, 1),
                )
                for x in ch
            )
        )
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

        if self.end2end:
            self.one2one_cv2 = copy.deepcopy(self.cv2)
            self.one2one_cv3 = copy.deepcopy(self.cv3)

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        if self.end2end:
            return self.forward_end2end(x)

        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:  # Training path
            return x
        y = self._inference(x)
        return y if self.export else (y, x)

    def forward_end2end(self, x):
        """
        Performs forward pass of the v10Detect module.

        Args:
            x (tensor): Input tensor.

        Returns:
            (dict, tensor): If not in training mode, returns a dictionary containing the outputs of both one2many and one2one detections.
                           If in training mode, returns a dictionary containing the outputs of one2many and one2one detections separately.
        """
        x_detach = [xi.detach() for xi in x]
        one2one = [
            torch.cat((self.one2one_cv2[i](x_detach[i]), self.one2one_cv3[i](x_detach[i])), 1) for i in range(self.nl)
        ]
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:  # Training path
            return {"one2many": x, "one2one": one2one}

        y = self._inference(one2one)
        y = self.postprocess(y.permute(0, 2, 1), self.max_det, self.nc)
        return y if self.export else (y, {"one2many": x, "one2one": one2one})

    def _inference(self, x):
        """Decode predicted bounding boxes and class probabilities based on multiple-level feature maps."""
        # Inference path
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.format != "imx" and (self.dynamic or self.shape != shape):
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        if self.export and self.format in {"saved_model", "pb", "tflite", "edgetpu", "tfjs"}:  # avoid TF FlexSplitV ops
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4 :]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

        if self.export and self.format in {"tflite", "edgetpu"}:
            # Precompute normalization factor to increase numerical stability
            # See https://github.com/ultralytics/ultralytics/issues/7371
            grid_h = shape[2]
            grid_w = shape[3]
            grid_size = torch.tensor([grid_w, grid_h, grid_w, grid_h], device=box.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * grid_size)
            dbox = self.decode_bboxes(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2])
        elif self.export and self.format == "imx":
            dbox = self.decode_bboxes(
                self.dfl(box) * self.strides, self.anchors.unsqueeze(0) * self.strides, xywh=False
            )
            return dbox.transpose(1, 2), cls.sigmoid().permute(0, 2, 1)
        else:
            dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides

        return torch.cat((dbox, cls.sigmoid()), 1)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)
        if self.end2end:
            for a, b, s in zip(m.one2one_cv2, m.one2one_cv3, m.stride):  # from
                a[-1].bias.data[:] = 1.0  # box
                b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)

    def decode_bboxes(self, bboxes, anchors, xywh=True):
        """Decode bounding boxes."""
        return dist2bbox(bboxes, anchors, xywh=xywh and (not self.end2end), dim=1)

    @staticmethod
    def postprocess(preds: torch.Tensor, max_det: int, nc: int = 80):
        """
        Post-processes YOLO model predictions.

        Args:
            preds (torch.Tensor): Raw predictions with shape (batch_size, num_anchors, 4 + nc) with last dimension
                format [x, y, w, h, class_probs].
            max_det (int): Maximum detections per image.
            nc (int, optional): Number of classes. Default: 80.

        Returns:
            (torch.Tensor): Processed predictions with shape (batch_size, min(max_det, num_anchors), 6) and last
                dimension format [x, y, w, h, max_class_prob, class_index].
        """
        batch_size, anchors, _ = preds.shape  # i.e. shape(16,8400,84)
        boxes, scores = preds.split([4, nc], dim=-1)
        index = scores.amax(dim=-1).topk(min(max_det, anchors))[1].unsqueeze(-1)
        boxes = boxes.gather(dim=1, index=index.repeat(1, 1, 4))
        scores = scores.gather(dim=1, index=index.repeat(1, 1, nc))
        scores, index = scores.flatten(1).topk(min(max_det, anchors))
        i = torch.arange(batch_size)[..., None]  # batch indices
        return torch.cat([boxes[i, index // nc], scores[..., None], (index % nc)[..., None].float()], dim=-1)


class _BudgetedContextBridge(nn.Module):
    """Per-channel residual-energy budget shared by CSTD cross-scale adapters."""

    def __init__(self, max_ratio, eps=1e-6, detach_budget=True):
        super().__init__()
        self.max_ratio, self.eps, self.detach_budget = float(max_ratio), float(eps), bool(detach_budget)
        if not 0.0 < self.max_ratio <= 1.0 or self.eps <= 0.0:
            raise ValueError("max_ratio must be in (0, 1] and eps must be positive.")

    def _budget(self, base, correction):
        if base.shape != correction.shape:
            raise ValueError("base and correction shapes differ.")
        base_energy = base.float().square().mean(dim=(2, 3), keepdim=True).add(self.eps).sqrt()
        correction_energy = correction.float().square().mean(dim=(2, 3), keepdim=True).add(self.eps).sqrt()
        scale = torch.minimum(torch.ones_like(correction_energy), self.max_ratio * base_energy / correction_energy.clamp_min(self.eps))
        if self.detach_budget:
            scale = scale.detach()
        return (correction.float() * scale).to(dtype=base.dtype)


class SemanticContextBridge(_BudgetedContextBridge):
    """Inject deeper semantic context into a shallower classification feature with a zero-start residual."""

    def __init__(self, c_target, c_context, max_ratio=0.08, reduction=4, eps=1e-6):
        super().__init__(max_ratio=max_ratio, eps=eps, detach_budget=True)
        self.c_target, self.c_context = int(c_target), int(c_context)
        if self.c_target <= 0 or self.c_context <= 0 or int(reduction) < 1:
            raise ValueError("SemanticContextBridge requires positive channels and reduction >= 1.")
        hidden = max(8, self.c_target // max(8, int(reduction) * 4))
        self.context_proj = Conv(self.c_context, self.c_target, 1, 1)
        self.gate = nn.Sequential(
            nn.Conv2d(2, hidden, 3, 1, 1, bias=True), nn.SiLU(inplace=True), nn.Conv2d(hidden, 1, 1, bias=True), nn.Sigmoid()
        )
        self.out_proj = nn.Conv2d(self.c_target, self.c_target, 1, bias=False)
        nn.init.zeros_(self.out_proj.weight)

    def forward(self, target, context):
        if target.ndim != 4 or context.ndim != 4 or target.shape[0] != context.shape[0]:
            raise ValueError("SemanticContextBridge expects compatible NCHW tensors.")
        if target.shape[1] != self.c_target or context.shape[1] != self.c_context:
            raise ValueError(f"expected target/context channels {self.c_target}/{self.c_context}.")
        context_feature = F.interpolate(self.context_proj(context), size=target.shape[-2:], mode="bilinear", align_corners=False)
        gate = self.gate(
            torch.cat((target.float().abs().mean(dim=1, keepdim=True), context_feature.float().abs().mean(dim=1, keepdim=True)), dim=1).to(target.dtype)
        )
        return target + self._budget(target, self.out_proj(context_feature * gate))


class BoundaryContextBridge(_BudgetedContextBridge):
    """Inject shallower boundary context into a deeper regression feature with a zero-start residual."""

    def __init__(self, c_target, c_source, max_ratio=0.08, reduction=4, eps=1e-6):
        super().__init__(max_ratio=max_ratio, eps=eps, detach_budget=True)
        self.c_target, self.c_source = int(c_target), int(c_source)
        if self.c_target <= 0 or self.c_source <= 0 or int(reduction) < 1:
            raise ValueError("BoundaryContextBridge requires positive channels and reduction >= 1.")
        hidden = max(self.c_target // int(reduction), 16)
        self.source_proj = Conv(self.c_source, self.c_target, 1, 1)
        sobel_x = torch.tensor(((-1.0, 0.0, 1.0), (-2.0, 0.0, 2.0), (-1.0, 0.0, 1.0)), dtype=torch.float32)
        self.register_buffer("sobel_x", sobel_x[None, None], persistent=False)
        self.register_buffer("sobel_y", sobel_x.t().contiguous()[None, None], persistent=False)
        self.edge_proj = nn.Sequential(
            nn.Conv2d(3, 8, 3, 1, 1, bias=True), nn.SiLU(inplace=True), nn.Conv2d(8, self.c_target, 1, bias=False)
        )
        self.mix = Conv(3 * self.c_target, hidden, 1, 1)
        self.out_proj = nn.Conv2d(hidden, self.c_target, 1, bias=False)
        nn.init.zeros_(self.out_proj.weight)

    def _edge_cues(self, source):
        summary = F.pad(source.float().mean(dim=1, keepdim=True), (1, 1, 1, 1), mode="replicate")
        grad_x = F.conv2d(summary, self.sobel_x.to(device=source.device, dtype=summary.dtype))
        grad_y = F.conv2d(summary, self.sobel_y.to(device=source.device, dtype=summary.dtype))
        rms = (grad_x.square() + grad_y.square()).mean(dim=(2, 3), keepdim=True).add(self.eps).sqrt()
        grad_x, grad_y = grad_x / rms, grad_y / rms
        magnitude = (grad_x.square() + grad_y.square() + self.eps).sqrt()
        return torch.cat((grad_x, grad_y, magnitude), dim=1).to(source.dtype)

    @staticmethod
    def _resize_down(feature, target_size):
        if feature.shape[-2:] == target_size:
            return feature
        if feature.shape[-2] == target_size[0] * 2 and feature.shape[-1] == target_size[1] * 2:
            return F.avg_pool2d(feature, kernel_size=2, stride=2)
        return F.interpolate(feature, size=target_size, mode="bilinear", align_corners=False)

    @staticmethod
    def _local_detail(target):
        low = F.avg_pool2d(F.pad(target, (1, 1, 1, 1), mode="replicate"), kernel_size=3, stride=1)
        return target - low

    def forward(self, target, source):
        if target.ndim != 4 or source.ndim != 4 or target.shape[0] != source.shape[0]:
            raise ValueError("BoundaryContextBridge expects compatible NCHW tensors.")
        if target.shape[1] != self.c_target or source.shape[1] != self.c_source:
            raise ValueError(f"expected target/source channels {self.c_target}/{self.c_source}.")
        target_size = target.shape[-2:]
        source_feature = self._resize_down(self.source_proj(source), target_size)
        edge_feature = self._resize_down(self.edge_proj(self._edge_cues(source)), target_size)
        correction = self.out_proj(self.mix(torch.cat((self._local_detail(target), source_feature, edge_feature), dim=1)))
        return target + self._budget(target, correction)


class CSTDDetect(Detect):
    """Cross-scale task-decoupled three-level Detect head preserving cv2/cv3/dfl checkpoint keys."""

    def __init__(self, nc=80, cls_residual_ratio=0.08, reg_residual_ratio=0.08, reduction=4, ch=()):
        if not isinstance(ch, (list, tuple)) or len(ch) != 3:
            raise ValueError(f"CSTDDetect requires P3/P4/P5 channels, got {ch}.")
        super().__init__(nc=nc, ch=ch)
        if self.end2end:
            raise NotImplementedError("CSTDDetect supports only the standard one-to-many path.")
        if self.nl != 3:
            raise ValueError(f"CSTDDetect requires exactly three levels, got {self.nl}.")
        with torch.random.fork_rng(devices=[], enabled=True):
            self.cls_p3_from_p4 = SemanticContextBridge(ch[0], ch[1], max_ratio=cls_residual_ratio, reduction=reduction)
            self.cls_p4_from_p5 = SemanticContextBridge(ch[1], ch[2], max_ratio=cls_residual_ratio, reduction=reduction)
            self.reg_p4_from_p3 = BoundaryContextBridge(ch[1], ch[0], max_ratio=reg_residual_ratio, reduction=reduction)
            self.reg_p5_from_p4 = BoundaryContextBridge(ch[2], ch[1], max_ratio=reg_residual_ratio, reduction=reduction)
        for bridge in (self.cls_p3_from_p4, self.cls_p4_from_p5, self.reg_p4_from_p3, self.reg_p5_from_p4):
            nn.init.zeros_(bridge.out_proj.weight)

    def forward(self, x):
        if not isinstance(x, (list, tuple)) or len(x) != self.nl:
            raise ValueError("CSTDDetect expects [P3, P4, P5].")
        p3, p4, p5 = x
        cls_features = (self.cls_p3_from_p4(p3, p4), self.cls_p4_from_p5(p4, p5), p5)
        reg_features = (p3, self.reg_p4_from_p3(p4, p3), self.reg_p5_from_p4(p5, p4))
        outputs = [torch.cat((self.cv2[index](reg_features[index]), self.cv3[index](cls_features[index])), dim=1) for index in range(self.nl)]
        if self.training:
            return outputs
        prediction = self._inference(outputs)
        return prediction if self.export else (prediction, outputs)


class _HRCTNode(nn.Module):
    """Reliability-gated target calibration using adjacent scales only as context cues."""

    def __init__(self, channels, reduction: int = 4, max_gain: float = 0.10, eps: float = 1e-6):
        super().__init__()
        if not isinstance(channels, (list, tuple)) or len(channels) < 2:
            raise ValueError("channels must contain one target and at least one context channel count.")
        if any(int(c) <= 0 for c in channels):
            raise ValueError(f"all channel counts must be positive, got {channels}.")
        if reduction < 1:
            raise ValueError(f"reduction must be at least 1, got {reduction}.")
        if not 0.0 < max_gain <= 1.0:
            raise ValueError(f"max_gain must be in (0, 1], got {max_gain}.")
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}.")

        channels = tuple(int(c) for c in channels)
        hidden = max(channels[0] // int(reduction), 16)
        gate_hidden = max(8, hidden // 4)
        cue_channels = 2 + 2 * (len(channels) - 1)
        self.channels = channels
        self.max_gain = float(max_gain)
        self.eps = float(eps)
        self.role = "unspecified"
        self.alpha_raw = nn.Parameter(torch.zeros(1, dtype=torch.float32))

        self.target_proj = nn.Conv2d(channels[0], hidden, 1, bias=True)
        self.context_proj = nn.ModuleList(nn.Conv2d(c, hidden, 1, bias=True) for c in channels[1:])
        self.delta = nn.Sequential(
            nn.Conv2d(channels[0], channels[0], 3, padding=1, groups=channels[0], bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.SiLU(),
            nn.Conv2d(channels[0], channels[0], 1, bias=True),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(cue_channels, gate_hidden, 3, padding=1, bias=True),
            nn.SiLU(),
            nn.Conv2d(gate_hidden, 1, 1, bias=True),
            nn.Sigmoid(),
        )

    def _normalize_cue(self, cue: torch.Tensor) -> torch.Tensor:
        cue = cue.float()
        cue = cue / (cue.mean(dim=(2, 3), keepdim=True) + self.eps)
        return cue.clamp(0.0, 4.0) * 0.25

    def forward(self, features) -> torch.Tensor:
        if not isinstance(features, (list, tuple)) or len(features) != len(self.channels):
            raise ValueError(f"expected {len(self.channels)} feature maps, got {type(features).__name__}.")
        target = features[0]
        if target.ndim != 4 or target.shape[1] != self.channels[0]:
            raise ValueError(f"invalid HRCT target shape {tuple(target.shape)} for channels={self.channels[0]}.")

        target_embed = self.target_proj(target)
        cues = [
            self._normalize_cue(target_embed.abs().mean(dim=1, keepdim=True)),
            self._normalize_cue(
                (target_embed.float() - target_embed.float().mean(dim=1, keepdim=True))
                .pow(2)
                .mean(dim=1, keepdim=True)
                .add(self.eps)
                .sqrt()
            ),
        ]
        target_normalized = F.normalize(target_embed.float(), dim=1, eps=self.eps)

        for index, (context, projection) in enumerate(zip(features[1:], self.context_proj), start=1):
            if context.ndim != 4 or context.shape[1] != self.channels[index]:
                raise ValueError(
                    f"invalid HRCT context shape {tuple(context.shape)} for channels={self.channels[index]}."
                )
            context = F.interpolate(context, size=target.shape[-2:], mode="bilinear", align_corners=False)
            context_embed = projection(context)
            context_normalized = F.normalize(context_embed.float(), dim=1, eps=self.eps)
            similarity = ((target_normalized * context_normalized).sum(dim=1, keepdim=True) + 1.0) * 0.5
            difference = (target_embed.float() - context_embed.float()).abs().mean(dim=1, keepdim=True)
            difference = difference / (difference.mean(dim=(2, 3), keepdim=True) + self.eps)
            cues.extend((similarity, difference.clamp(0.0, 4.0) * 0.25))

        reliability_gate = self.gate(torch.cat(cues, dim=1).to(dtype=target.dtype))
        gain = (self.max_gain * torch.tanh(self.alpha_raw)).to(dtype=target.dtype)
        return target + gain * reliability_gate * self.delta(target)


class HRCTDetect(Detect):
    """Detect head with identity-initialized hierarchical reliability calibration."""

    def __init__(self, nc=80, reduction=4, p3_gain=0.10, p4_gain=0.06, p5_gain=0.025, ch=()):
        if len(ch) != 3:
            raise ValueError("HRCTDetect requires exactly three detection scales: P3, P4 and P5.")
        super().__init__(nc=nc, ch=ch)
        self.hrct_p3 = self._make_node((ch[0], ch[1]), reduction, p3_gain, "detail")
        self.hrct_p4 = self._make_node((ch[1], ch[0], ch[2]), reduction, p4_gain, "balanced")
        self.hrct_p5 = self._make_node((ch[2], ch[1]), reduction, p5_gain, "semantic")

    @staticmethod
    def _make_node(channels, reduction, gain, role):
        if gain <= 0:
            return None
        node = _HRCTNode(channels=channels, reduction=reduction, max_gain=gain)
        node.role = role
        return node

    def forward(self, x):
        if not isinstance(x, (list, tuple)) or len(x) != 3:
            raise ValueError("HRCTDetect expects a three-element P3/P4/P5 feature sequence.")
        raw = tuple(x)
        p3 = raw[0] if self.hrct_p3 is None else self.hrct_p3([raw[0], raw[1]])
        p4 = raw[1] if self.hrct_p4 is None else self.hrct_p4([raw[1], raw[0], raw[2]])
        p5 = raw[2] if self.hrct_p5 is None else self.hrct_p5([raw[2], raw[1]])
        return super().forward([p3, p4, p5])


class NonUniformDFL(nn.Module):
    """Distribution focal projection with monotonically spaced non-uniform bins."""

    def __init__(self, reg_max: int = 16, gamma: float = 1.5):
        super().__init__()
        if reg_max <= 1:
            raise ValueError(f"reg_max must be greater than 1, got {reg_max}.")
        if gamma < 1.0:
            raise ValueError(f"gamma must be at least 1.0, got {gamma}.")

        index = torch.arange(reg_max, dtype=torch.float32)
        project = (index / (reg_max - 1)).pow(float(gamma)) * (reg_max - 1)
        project[0] = 0.0
        project[-1] = float(reg_max - 1)
        if not torch.all(project[1:] > project[:-1]):
            raise ValueError("non-uniform DFL projection must be strictly increasing.")

        self.reg_max = int(reg_max)
        self.gamma = float(gamma)
        self.register_buffer("project", project, persistent=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"NonUniformDFL expected [B, 4 * reg_max, A], got {tuple(x.shape)}.")
        b, c, a = x.shape
        if not torch.jit.is_tracing() and c != 4 * self.reg_max:
            raise ValueError(f"expected {4 * self.reg_max} channels, got {c}.")
        probability = x.view(b, 4, self.reg_max, a).softmax(dim=2)
        project = self.project.to(device=x.device, dtype=probability.dtype).view(1, 1, self.reg_max, 1)
        return (probability * project).sum(dim=2)


class SUDLDetect(HRCTDetect):
    """HRCT detection head using non-uniform distribution focal projection."""

    def __init__(
        self,
        nc=80,
        reduction=4,
        p3_gain=0.10,
        p4_gain=0.06,
        p5_gain=0.025,
        dfl_gamma=1.5,
        ch=(),
    ):
        super().__init__(
            nc=nc,
            reduction=reduction,
            p3_gain=p3_gain,
            p4_gain=p4_gain,
            p5_gain=p5_gain,
            ch=ch,
        )
        self.dfl = NonUniformDFL(self.reg_max, gamma=dfl_gamma)
        self.sudl_enabled = True
        self.dfl_gamma = float(dfl_gamma)


class SBRHDetect(Detect):
    """Side-boundary resampling head that refines the original DFL logits."""

    unsupported_refine_formats = {"saved_model", "pb", "tflite", "edgetpu", "tfjs", "imx", "coreml"}

    def __init__(
        self,
        nc=80,
        refine_ratio=1.0,
        max_sample_offset=12.0,
        confidence_floor=0.25,
        max_refine_gain=0.10,
        detach_distance=True,
        ch=(),
    ):
        super().__init__(nc=nc, ch=ch)
        if self.end2end:
            raise NotImplementedError("SBRHDetect only supports the standard one-to-many Detect path.")
        if self.nl != 3:
            raise ValueError(f"SBRHDetect requires exactly P3/P4/P5 inputs, got nl={self.nl}.")
        if float(refine_ratio) <= 0.0:
            raise ValueError(f"refine_ratio must be positive, got {refine_ratio}.")
        if not 0.0 < float(max_sample_offset) <= float(self.reg_max - 1):
            raise ValueError(f"max_sample_offset must be in (0, {self.reg_max - 1}], got {max_sample_offset}.")
        if not 0.0 <= float(confidence_floor) <= 1.0:
            raise ValueError(f"confidence_floor must be in [0, 1], got {confidence_floor}.")
        if float(max_refine_gain) < 0.0:
            raise ValueError(f"max_refine_gain must be non-negative, got {max_refine_gain}.")
        if any(len(tower) != 3 for tower in self.cv2):
            raise RuntimeError("SBRHDetect expects every Detect.cv2 tower to contain exactly 3 layers.")

        reg_channels = int(self.cv2[0][-1].in_channels)
        if any(int(tower[-1].in_channels) != reg_channels for tower in self.cv2):
            raise RuntimeError("SBRHDetect requires shared regression widths at all scales.")
        hidden = max(int(round(reg_channels * float(refine_ratio))), 16)
        self.reg_channels = reg_channels
        self.max_sample_offset = float(max_sample_offset)
        self.confidence_floor = float(confidence_floor)
        self.max_refine_gain = float(max_refine_gain)
        self.detach_distance = bool(detach_distance)
        self.side_refiner = nn.Sequential(
            Conv(4 * reg_channels, hidden, 1, 1),
            Conv(hidden, hidden, 3, 1, g=hidden),
            Conv(hidden, hidden, 1, 1),
            nn.Conv2d(hidden, self.reg_max, 1, bias=True),
        )
        nn.init.normal_(self.side_refiner[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.side_refiner[-1].bias)
        self.side_alpha_raw = nn.Parameter(torch.zeros(4, dtype=torch.float32))
        self.register_buffer("boundary_project", torch.arange(self.reg_max, dtype=torch.float32), persistent=False)

    @staticmethod
    def _base_grid(batch, height, width, device):
        """Return pixel-center coordinates for align_corners=False sampling."""
        y = (torch.arange(height, device=device, dtype=torch.float32) + 0.5) * (2.0 / max(height, 1)) - 1.0
        x = (torch.arange(width, device=device, dtype=torch.float32) + 0.5) * (2.0 / max(width, 1)) - 1.0
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        return torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0).expand(batch, -1, -1, -1)

    def _sampling_distance(self, coarse_logits):
        """Entropy-shrink coarse LTRB expectations before boundary sampling."""
        batch, channels, height, width = coarse_logits.shape
        if channels != 4 * self.reg_max:
            raise ValueError(f"Expected {4 * self.reg_max} box channels, got {channels}.")
        probability = coarse_logits.float().view(batch, 4, self.reg_max, height, width).softmax(dim=2)
        project = self.boundary_project.to(device=probability.device, dtype=probability.dtype).view(1, 1, -1, 1, 1)
        expectation = (probability * project).sum(dim=2)
        entropy = -(probability.clamp_min(1e-8).log() * probability).sum(dim=2) / math.log(self.reg_max)
        confidence = self.confidence_floor + (1.0 - self.confidence_floor) * (1.0 - entropy.clamp(0.0, 1.0))
        distance = (expectation * confidence).clamp(0.0, self.max_sample_offset)
        return distance.detach() if self.detach_distance else distance

    def _sample_boundaries(self, feature, distance):
        """Vectorized FP32 grid sampling for left/top/right/bottom positions."""
        batch, channels, height, width = feature.shape
        zero = torch.zeros_like(distance[:, 0])
        offset_x = torch.stack((-distance[:, 0], zero, distance[:, 2], zero), dim=1)
        offset_y = torch.stack((zero, -distance[:, 1], zero, distance[:, 3]), dim=1)
        base = self._base_grid(batch, height, width, feature.device).unsqueeze(1)
        grid = base + torch.stack(
            (offset_x.float() * (2.0 / max(width, 1)), offset_y.float() * (2.0 / max(height, 1))), dim=-1
        )
        source = feature.float().unsqueeze(1).expand(-1, 4, -1, -1, -1).reshape(batch * 4, channels, height, width)
        sampled = F.grid_sample(
            source, grid.reshape(batch * 4, height, width, 2), mode="bilinear", padding_mode="border", align_corners=False
        )
        return sampled.reshape(batch, 4, channels, height, width).to(dtype=feature.dtype)

    def _refine_box_logits(self, regression_feature, coarse_logits):
        batch, channels, height, width = regression_feature.shape
        boundary = self._sample_boundaries(regression_feature, self._sampling_distance(coarse_logits))
        center = regression_feature.unsqueeze(1).expand(-1, 4, -1, -1, -1)
        signed_difference = boundary - center
        refiner_input = torch.cat(
            (center, boundary, signed_difference, signed_difference.abs()), dim=2
        ).reshape(batch * 4, 4 * channels, height, width)
        residual_logits = self.side_refiner(refiner_input).reshape(batch, 4, self.reg_max, height, width)
        coarse = coarse_logits.view(batch, 4, self.reg_max, height, width)
        gain = self.max_refine_gain * torch.tanh(self.side_alpha_raw).view(1, 4, 1, 1, 1)
        return (coarse + gain * residual_logits).reshape(batch, 4 * self.reg_max, height, width)

    def forward(self, x):
        if not isinstance(x, (list, tuple)) or len(x) != self.nl:
            count = len(x) if isinstance(x, (list, tuple)) else type(x).__name__
            raise ValueError(f"SBRHDetect expects {self.nl} feature maps, got {count}.")
        skip_refinement = self.export and self.format in self.unsupported_refine_formats
        outputs = []
        for index in range(self.nl):
            regression_feature = self.cv2[index][0](x[index])
            regression_feature = self.cv2[index][1](regression_feature)
            coarse_logits = self.cv2[index][2](regression_feature)
            box_logits = coarse_logits if skip_refinement else self._refine_box_logits(regression_feature, coarse_logits)
            outputs.append(torch.cat((box_logits, self.cv3[index](x[index])), dim=1))
        if self.training:
            return outputs
        prediction = self._inference(outputs)
        return prediction if self.export else (prediction, outputs)




class P3TaskAdapter(nn.Module):
    """Identity-initialized task-specific residual adapter for the P3 feature only."""

    def __init__(self, channels, max_gain=0.10, reduction=4):
        super().__init__()
        self.channels, self.max_gain = int(channels), float(max_gain)
        if self.channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}.")
        if self.max_gain < 0.0 or int(reduction) < 1:
            raise ValueError("max_gain must be non-negative and reduction must be >= 1.")

        self.local_branch = nn.Sequential(
            nn.Conv2d(self.channels, self.channels, 3, 1, 1, groups=self.channels, bias=False),
            nn.BatchNorm2d(self.channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(self.channels, self.channels, 1, bias=True),
        )
        self.horizontal = nn.Conv2d(self.channels, self.channels, (1, 5), 1, (0, 2), groups=self.channels, bias=False)
        self.vertical = nn.Conv2d(self.channels, self.channels, (5, 1), 1, (2, 0), groups=self.channels, bias=False)
        self.boundary_norm = nn.BatchNorm2d(self.channels)
        self.boundary_act = nn.SiLU(inplace=True)
        hidden = max(self.channels // int(reduction), 16)
        self.task_gate = nn.Sequential(
            nn.Conv2d(3, hidden, 3, 1, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 2, 1, bias=True),
            nn.Sigmoid(),
        )
        self.reg_out = nn.Conv2d(self.channels, self.channels, 1, bias=True)
        self.cls_out = nn.Conv2d(self.channels, self.channels, 1, bias=True)
        for projection in (self.reg_out, self.cls_out):
            nn.init.kaiming_normal_(projection.weight, mode="fan_out", nonlinearity="linear")
            nn.init.zeros_(projection.bias)
        # [regression, classification], zero-start to exactly preserve Detect at initialization.
        self.alpha_raw = nn.Parameter(torch.zeros(2, dtype=torch.float32))

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError(f"expected [B,{self.channels},H,W], got {tuple(x.shape)}.")
        local = self.local_branch(x)
        boundary = self.boundary_act(self.boundary_norm(self.horizontal(x) + self.vertical(x)))
        gate = self.task_gate(
            torch.cat(
                (
                    x.abs().mean(dim=1, keepdim=True),
                    local.abs().mean(dim=1, keepdim=True),
                    boundary.abs().mean(dim=1, keepdim=True),
                ),
                dim=1,
            )
        )
        gains = self.max_gain * torch.tanh(self.alpha_raw)
        regression = x + gains[0] * self.reg_out(boundary * gate[:, :1])
        classification = x + gains[1] * self.cls_out(local * gate[:, 1:2])
        return regression, classification


class P3DecoupledDetect(Detect):
    """Original Detect output contract with task adaptation only for the P3 input."""

    def __init__(self, nc=80, max_gain=0.10, reduction=4, ch=()):
        super().__init__(nc=nc, ch=ch)
        if self.end2end:
            raise NotImplementedError("P3DecoupledDetect supports only the standard Detect path.")
        if self.nl != 3 or len(ch) != 3:
            raise ValueError(f"P3DecoupledDetect requires P3/P4/P5, got channels={ch}.")
        self.p3_adapter = P3TaskAdapter(channels=ch[0], max_gain=max_gain, reduction=reduction)

    def forward(self, x):
        if not isinstance(x, (list, tuple)) or len(x) != self.nl:
            raise ValueError(f"P3DecoupledDetect expects {self.nl} feature maps.")
        features = list(x)
        p3_regression, p3_classification = self.p3_adapter(features[0])
        outputs = []
        for index in range(self.nl):
            regression_input = p3_regression if index == 0 else features[index]
            classification_input = p3_classification if index == 0 else features[index]
            outputs.append(torch.cat((self.cv2[index](regression_input), self.cv3[index](classification_input)), dim=1))
        if self.training:
            return outputs
        prediction = self._inference(outputs)
        return prediction if self.export else (prediction, outputs)


class HighResolutionEvidenceEncoder(nn.Module):
    """Encode anti-aliased P2 evidence at the final P3 spatial resolution."""

    def __init__(self, c_shallow, c_p3, reduction=4, eps=1e-6):
        super().__init__()
        self.c_shallow, self.c_p3, self.eps = int(c_shallow), int(c_p3), float(eps)
        reduction = int(reduction)
        if self.c_shallow <= 0 or self.c_p3 <= 0:
            raise ValueError(f"Channels must be positive, got {self.c_shallow}->{self.c_p3}.")
        if reduction < 1 or self.eps <= 0.0:
            raise ValueError(f"reduction and eps must be positive, got {reduction}, {self.eps}.")

        self.evidence_channels = max(self.c_p3 // 4, 16)
        hidden = max(self.c_p3 // reduction, 16)
        blur = torch.tensor(((1.0, 2.0, 1.0), (2.0, 4.0, 2.0), (1.0, 2.0, 1.0)), dtype=torch.float32)
        self.register_buffer("blur_kernel", (blur / blur.sum())[None, None], persistent=False)
        self.low_proj = Conv(self.c_shallow, self.evidence_channels, 1, 1)
        self.contrast_proj = Conv(self.c_shallow, self.evidence_channels, 1, 1)
        self.fuse = nn.Sequential(
            Conv(2 * self.evidence_channels, hidden, 3, 1),
            Conv(hidden, self.evidence_channels, 1, 1),
        )

    @staticmethod
    def _zscore_map(x, eps):
        """Return a per-image spatial z-score with a safe zero-variance path."""
        x_float = x.float()
        mean = x_float.mean(dim=(2, 3), keepdim=True)
        std = x_float.var(dim=(2, 3), keepdim=True, unbiased=False).add(eps).sqrt()
        return ((x_float - mean) / std).to(dtype=x.dtype)

    def _blur_downsample(self, x, target_size):
        channels = x.shape[1]
        kernel = self.blur_kernel.to(device=x.device, dtype=x.dtype).repeat(channels, 1, 1, 1)
        output = F.conv2d(F.pad(x, (1, 1, 1, 1), mode="replicate"), kernel, stride=2, groups=channels)
        return F.interpolate(output, size=target_size, mode="bilinear", align_corners=False) if output.shape[-2:] != target_size else output

    def forward(self, shallow, target_size):
        if shallow.ndim != 4 or shallow.shape[1] != self.c_shallow:
            raise ValueError(f"Expected NCHW shallow tensor with channels={self.c_shallow}, got {tuple(shallow.shape)}.")
        if not isinstance(target_size, (tuple, list)) or len(target_size) != 2:
            raise ValueError(f"target_size must be (height, width), got {target_size}.")
        target_size = tuple(map(int, target_size))
        if min(target_size) <= 0:
            raise ValueError(f"target_size must be positive, got {target_size}.")

        low_raw = self._blur_downsample(shallow, target_size)
        average_raw = F.adaptive_avg_pool2d(shallow, target_size)
        contrast_raw = (F.adaptive_max_pool2d(shallow, target_size) - average_raw).clamp_min(0.0)
        evidence = self.fuse(torch.cat((self.low_proj(low_raw), self.contrast_proj(contrast_raw)), dim=1))
        raw_energy = contrast_raw.float().mean(dim=1, keepdim=True)
        prior = torch.sigmoid(self._zscore_map(raw_energy, self.eps))
        prior = (0.5 * prior + 0.5 * F.avg_pool2d(prior, kernel_size=3, stride=1, padding=1)).clamp(0.0, 1.0)
        return evidence, prior.to(dtype=evidence.dtype)


class AmbiguityReactivationGate(nn.Module):
    """Activate P2-supported locations whose final P3 response is relatively weak."""

    def __init__(self, evidence_channels, c_p3, reduction=4, eps=1e-6):
        super().__init__()
        self.evidence_channels, self.c_p3, self.eps = int(evidence_channels), int(c_p3), float(eps)
        reduction = int(reduction)
        if self.evidence_channels <= 0 or self.c_p3 <= 0:
            raise ValueError("evidence_channels and c_p3 must be positive.")
        if reduction < 1 or self.eps <= 0.0:
            raise ValueError(f"reduction and eps must be positive, got {reduction}, {self.eps}.")

        hidden = max(self.c_p3 // reduction, 16)
        self.semantic_proj = Conv(self.c_p3, self.evidence_channels, 1, 1)
        self.compatibility = nn.Sequential(
            Conv(2 * self.evidence_channels, hidden, 3, 1),
            Conv(hidden, hidden, 3, 1, g=hidden),
            nn.Conv2d(hidden, 1, 1, bias=True),
        )
        nn.init.zeros_(self.compatibility[-1].weight)
        nn.init.zeros_(self.compatibility[-1].bias)

    @staticmethod
    def _zscore_map(x, eps):
        x_float = x.float()
        mean = x_float.mean(dim=(2, 3), keepdim=True)
        std = x_float.var(dim=(2, 3), keepdim=True, unbiased=False).add(eps).sqrt()
        return ((x_float - mean) / std).to(dtype=x.dtype)

    def forward(self, evidence, evidence_prior, p3):
        if evidence.ndim != 4 or evidence_prior.ndim != 4 or p3.ndim != 4:
            raise ValueError("evidence, evidence_prior and p3 must be NCHW tensors.")
        if evidence.shape[0] != p3.shape[0] or evidence.shape[1] != self.evidence_channels or p3.shape[1] != self.c_p3:
            raise ValueError("Unexpected batch size or channel count in ambiguity gate.")
        if evidence_prior.shape[:2] != (p3.shape[0], 1) or evidence.shape[-2:] != p3.shape[-2:] or evidence_prior.shape[-2:] != p3.shape[-2:]:
            raise ValueError("evidence, evidence_prior and p3 must share the required NCHW shape.")

        semantic = self.semantic_proj(p3)
        compatibility = torch.sigmoid(self.compatibility(torch.cat((evidence, semantic), dim=1)))
        semantic_energy = p3.float().abs().mean(dim=1, keepdim=True)
        semantic_confidence = torch.sigmoid(self._zscore_map(semantic_energy, self.eps)).to(dtype=p3.dtype)
        ambiguity = evidence_prior.detach() * (1.0 - semantic_confidence.detach()) * compatibility
        return ambiguity.clamp(0.0, 1.0)


class P2RecallReactivation(nn.Module):
    """Positive-only spatial-channel P2 evidence reactivation for P3 classification."""

    def __init__(
        self, c_shallow, c_p3, max_gain=0.10, reduction=4, gain_init=-4.0, use_ambiguity=True, use_channel=True, eps=1e-6
    ):
        super().__init__()
        self.c_shallow, self.c_p3, self.max_gain = int(c_shallow), int(c_p3), float(max_gain)
        reduction, gain_init, eps = int(reduction), float(gain_init), float(eps)
        if self.c_shallow <= 0 or self.c_p3 <= 0:
            raise ValueError(f"Channels must be positive, got {self.c_shallow}->{self.c_p3}.")
        if self.max_gain < 0.0 or reduction < 1 or eps <= 0.0:
            raise ValueError(f"Invalid max_gain/reduction/eps: {self.max_gain}, {reduction}, {eps}.")
        self.use_ambiguity, self.use_channel = bool(use_ambiguity), bool(use_channel)
        self.encoder = HighResolutionEvidenceEncoder(self.c_shallow, self.c_p3, reduction=reduction, eps=eps)
        self.ambiguity_gate = AmbiguityReactivationGate(self.encoder.evidence_channels, self.c_p3, reduction=reduction, eps=eps)
        self.channel_gate = nn.Conv2d(self.encoder.evidence_channels, self.c_p3, 1, bias=True)
        nn.init.zeros_(self.channel_gate.weight)
        nn.init.zeros_(self.channel_gate.bias)
        self.gain_raw = nn.Parameter(torch.tensor(gain_init, dtype=torch.float32))

    def forward(self, shallow, p3):
        if shallow.ndim != 4 or p3.ndim != 4 or shallow.shape[0] != p3.shape[0]:
            raise ValueError("P2RecallReactivation expects two same-batch NCHW tensors.")
        if shallow.shape[1] != self.c_shallow or p3.shape[1] != self.c_p3:
            raise ValueError(f"Expected channels {self.c_shallow}->{self.c_p3}, got {shallow.shape[1]}->{p3.shape[1]}.")
        evidence, evidence_prior = self.encoder(shallow, p3.shape[-2:])
        spatial_gate = self.ambiguity_gate(evidence, evidence_prior, p3) if self.use_ambiguity else evidence_prior.detach()
        channel_gate = torch.sigmoid(self.channel_gate(evidence)) if self.use_channel else torch.ones_like(p3)
        gain = (self.max_gain * torch.sigmoid(self.gain_raw)).to(dtype=p3.dtype)
        multiplier = 1.0 + gain * spatial_gate * channel_gate
        return p3 * multiplier, spatial_gate


class RAMPDetect(Detect):
    """Standard three-scale Detect head with P2-guided P3 classification reactivation."""

    def __init__(
        self, nc=80, max_gain=0.10, reduction=4, gain_init=-4.0, use_ambiguity=True, use_channel=True, c_shallow=0, ch=()
    ):
        if not isinstance(ch, (list, tuple)) or len(ch) != 3:
            raise ValueError(f"RAMPDetect requires P3/P4/P5 channels, got {ch}.")
        c_shallow = int(c_shallow)
        if c_shallow <= 0:
            raise ValueError(f"c_shallow must be positive, got {c_shallow}.")
        super().__init__(nc=nc, ch=ch)
        if self.end2end:
            raise NotImplementedError("RAMPDetect supports only the standard one-to-many path.")
        if self.nl != 3:
            raise ValueError(f"RAMPDetect requires exactly 3 detection levels, got {self.nl}.")
        self.c_shallow = c_shallow
        self.p3_reactivation = P2RecallReactivation(
            c_shallow=c_shallow,
            c_p3=int(ch[0]),
            max_gain=max_gain,
            reduction=reduction,
            gain_init=gain_init,
            use_ambiguity=use_ambiguity,
            use_channel=use_channel,
        )

    def forward(self, x):
        if not isinstance(x, (list, tuple)) or len(x) != 4:
            raise ValueError("RAMPDetect expects [P2, P3, P4, P5].")
        shallow, p3, p4, p5 = x
        if shallow.shape[1] != self.c_shallow:
            raise ValueError(f"Expected P2 channels={self.c_shallow}, got {shallow.shape[1]}.")
        p3_classification, _ = self.p3_reactivation(shallow, p3)
        features, outputs = (p3, p4, p5), []
        for index, feature in enumerate(features):
            class_input = p3_classification if index == 0 else feature
            outputs.append(torch.cat((self.cv2[index](feature), self.cv3[index](class_input)), dim=1))
        if self.training:
            return outputs
        prediction = self._inference(outputs)
        return prediction if self.export else (prediction, outputs)


class GradientIsolatedShallowEncoder(nn.Module):
    """Encode detached, anti-aliased P2 evidence using detached P3 semantic context."""

    def __init__(self, c_shallow, c_p3, c_hidden, reduction=4, detach_guidance=True, eps=1e-6):
        super().__init__()
        self.c_shallow, self.c_p3, self.c_hidden = int(c_shallow), int(c_p3), int(c_hidden)
        reduction, self.eps = int(reduction), float(eps)
        if min(self.c_shallow, self.c_p3, self.c_hidden) <= 0:
            raise ValueError("c_shallow, c_p3 and c_hidden must all be positive.")
        if reduction < 1 or self.eps <= 0.0:
            raise ValueError(f"reduction and eps must be positive, got {reduction}, {self.eps}.")
        self.detach_guidance = bool(detach_guidance)
        self.embed_channels = max(self.c_hidden // reduction, 16)
        blur = torch.tensor(((1.0, 2.0, 1.0), (2.0, 4.0, 2.0), (1.0, 2.0, 1.0)), dtype=torch.float32)
        self.register_buffer("blur_kernel", (blur / blur.sum())[None, None], persistent=False)
        self.low_proj = Conv(self.c_shallow, self.embed_channels, 1, 1)
        self.contrast_proj = Conv(self.c_shallow, self.embed_channels, 1, 1)
        self.semantic_proj = Conv(self.c_p3, self.embed_channels, 1, 1)
        self.fuse = nn.Sequential(
            Conv(4 * self.embed_channels, self.c_hidden, 3, 1),
            DWConv(self.c_hidden, self.c_hidden, 3, 1),
            Conv(self.c_hidden, self.c_hidden, 1, 1),
        )

    @staticmethod
    def _spatial_zscore(value, eps):
        value = value.float()
        mean = value.mean(dim=(2, 3), keepdim=True)
        std = value.var(dim=(2, 3), keepdim=True, unbiased=False).add(eps).sqrt()
        return (value - mean) / std

    def _blur_downsample(self, shallow, target_size):
        channels = shallow.shape[1]
        kernel = self.blur_kernel.to(device=shallow.device, dtype=shallow.dtype).repeat(channels, 1, 1, 1)
        output = F.conv2d(F.pad(shallow, (1, 1, 1, 1), mode="replicate"), kernel, stride=2, padding=0, groups=channels)
        return F.interpolate(output, size=target_size, mode="bilinear", align_corners=False) if output.shape[-2:] != target_size else output

    def forward(self, shallow, p3):
        if shallow.ndim != 4 or p3.ndim != 4:
            raise ValueError("GradientIsolatedShallowEncoder expects two NCHW tensors.")
        if shallow.shape[0] != p3.shape[0]:
            raise ValueError("P2 and P3 must have the same batch size.")
        if shallow.shape[1] != self.c_shallow or p3.shape[1] != self.c_p3:
            raise ValueError(f"Expected P2/P3 channels {self.c_shallow}/{self.c_p3}, got {shallow.shape[1]}/{p3.shape[1]}.")
        shallow_source, p3_source = (shallow.detach(), p3.detach()) if self.detach_guidance else (shallow, p3)
        target_size = p3.shape[-2:]
        low_raw = self._blur_downsample(shallow_source, target_size)
        average_raw = F.adaptive_avg_pool2d(shallow_source, target_size)
        contrast_raw = (F.adaptive_max_pool2d(shallow_source, target_size) - average_raw).clamp_min(0.0)
        low, contrast, semantic = self.low_proj(low_raw), self.contrast_proj(contrast_raw), self.semantic_proj(p3_source)
        shallow_combined = low + contrast
        auxiliary_hidden = self.fuse(torch.cat((low, contrast, semantic, (shallow_combined - semantic).abs()), dim=1))

        contrast_energy = contrast_raw.float().abs().mean(dim=1, keepdim=True)
        low_energy = low_raw.float().abs().mean(dim=1, keepdim=True)
        relative_confidence = torch.sigmoid(self._spatial_zscore(contrast_energy, self.eps))
        snr = (contrast_energy / (low_energy + self.eps)).clamp(0.0, 8.0)
        absolute_confidence = 1.0 - torch.exp(-snr)
        prior = relative_confidence * absolute_confidence
        prior = (0.75 * prior + 0.25 * F.avg_pool2d(prior, kernel_size=3, stride=1, padding=1)).clamp(0.0, 1.0)
        return auxiliary_hidden, prior.to(dtype=auxiliary_hidden.dtype)


class ClassPrototypeComplementaryRecovery(nn.Module):
    """Recover class-specific P3 misses without routing auxiliary gradients into backbone/classifier weights."""

    def __init__(
        self,
        c_shallow,
        c_p3,
        c_hidden,
        nc,
        reg_max=16,
        max_delta=1.50,
        loc_floor=0.50,
        support_self=0.75,
        miss_power=1.0,
        reduction=4,
        candidate_bias=-2.0,
        gain_init=-2.2,
        use_spatial_prior=True,
        use_class_gate=True,
        use_loc_guard=True,
        detach_guidance=True,
        eps=1e-6,
    ):
        super().__init__()
        self.c_shallow, self.c_p3, self.c_hidden, self.nc = map(int, (c_shallow, c_p3, c_hidden, nc))
        self.reg_max, reduction = int(reg_max), int(reduction)
        self.max_delta, self.loc_floor = float(max_delta), float(loc_floor)
        self.support_self, self.miss_power = float(support_self), float(miss_power)
        self.candidate_bias_init, self.eps = float(candidate_bias), float(eps)
        if min(self.c_shallow, self.c_p3, self.c_hidden, self.nc) <= 0:
            raise ValueError("All channel and class counts must be positive.")
        if self.reg_max <= 1 or reduction < 1 or self.eps <= 0.0:
            raise ValueError("reg_max must be > 1; reduction and eps must be positive.")
        if self.max_delta < 0.0 or not 0.0 <= self.loc_floor <= 1.0 or not 0.0 <= self.support_self <= 1.0 or self.miss_power < 0.0:
            raise ValueError("Invalid CPCR bounded-recovery arguments.")
        self.use_spatial_prior, self.use_class_gate, self.use_loc_guard = bool(use_spatial_prior), bool(use_class_gate), bool(use_loc_guard)
        self.encoder = GradientIsolatedShallowEncoder(
            self.c_shallow, self.c_p3, self.c_hidden, reduction=reduction, detach_guidance=detach_guidance, eps=self.eps
        )
        gate_hidden = max(self.c_hidden // reduction, 16)
        self.class_gate = nn.Sequential(
            Conv(3 * self.c_hidden + 1, gate_hidden, 3, 1),
            DWConv(gate_hidden, gate_hidden, 3, 1),
            nn.Conv2d(gate_hidden, self.nc, 1, 1, 0, bias=True),
        )
        nn.init.normal_(self.class_gate[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.class_gate[-1].bias)
        self.candidate_bias = nn.Parameter(torch.full((self.nc,), self.candidate_bias_init, dtype=torch.float32))
        self.gain_raw = nn.Parameter(torch.full((self.nc,), float(gain_init), dtype=torch.float32))

    def _localization_guard(self, box_logits):
        if box_logits.ndim != 4:
            raise ValueError("box_logits must be an NCHW tensor.")
        batch, channels, height, width = box_logits.shape
        expected = 4 * self.reg_max
        if channels != expected:
            raise ValueError(f"Expected {expected} DFL channels, got {channels}.")
        probability = box_logits.float().view(batch, 4, self.reg_max, height, width).softmax(dim=2)
        entropy = -(probability.clamp_min(self.eps).log() * probability).sum(dim=2) / math.log(self.reg_max)
        confidence = (1.0 - entropy.mean(dim=1, keepdim=True)).clamp(0.0, 1.0)
        guard = self.loc_floor + (1.0 - self.loc_floor) * confidence
        return guard.detach().to(dtype=box_logits.dtype)

    def _prototype_logits(self, auxiliary_hidden, classifier_conv):
        if not isinstance(classifier_conv, nn.Conv2d):
            raise TypeError("classifier_conv must be the final nn.Conv2d of cv3[0].")
        if classifier_conv.kernel_size != (1, 1) or classifier_conv.groups != 1:
            raise ValueError("The reused classifier convolution must be an ungrouped 1x1 convolution.")
        if classifier_conv.in_channels != self.c_hidden or classifier_conv.out_channels != self.nc:
            raise ValueError("Classifier channels do not match CPCR hidden/class dimensions.")
        return F.conv2d(
            auxiliary_hidden,
            classifier_conv.weight.detach(),
            bias=self.candidate_bias.to(dtype=auxiliary_hidden.dtype),
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
        )

    def forward(self, shallow, p3, base_hidden, base_logits, box_logits, classifier_conv):
        if base_hidden.ndim != 4 or base_logits.ndim != 4:
            raise ValueError("base_hidden and base_logits must be NCHW tensors.")
        if base_hidden.shape[1] != self.c_hidden or base_logits.shape[1] != self.nc:
            raise ValueError("Unexpected CPCR base hidden or class-logit channel count.")
        auxiliary_hidden, spatial_prior = self.encoder(shallow, p3)
        if auxiliary_hidden.shape != base_hidden.shape:
            raise ValueError(f"Auxiliary/base hidden shapes must match, got {tuple(auxiliary_hidden.shape)} and {tuple(base_hidden.shape)}.")
        candidate_logits = self._prototype_logits(auxiliary_hidden, classifier_conv)
        base_hidden_detached = base_hidden.detach()
        gate_input = torch.cat((auxiliary_hidden, base_hidden_detached, (auxiliary_hidden - base_hidden_detached).abs(), spatial_prior), dim=1)
        class_gate = torch.sigmoid(self.class_gate(gate_input)) if self.use_class_gate else torch.ones_like(candidate_logits)
        candidate_probability = torch.sigmoid(candidate_logits)
        if self.support_self < 1.0:
            neighborhood_probability = F.avg_pool2d(candidate_probability, kernel_size=3, stride=1, padding=1)
            local_support = torch.exp(
                self.support_self * torch.log(candidate_probability.clamp_min(self.eps))
                + (1.0 - self.support_self) * torch.log(neighborhood_probability.clamp_min(self.eps))
            )
        else:
            local_support = candidate_probability
        missing_condition = (1.0 - torch.sigmoid(base_logits.detach())).pow(self.miss_power)
        prior = spatial_prior.detach() if self.use_spatial_prior else torch.ones_like(spatial_prior)
        localization_guard = self._localization_guard(box_logits) if self.use_loc_guard else torch.ones_like(spatial_prior)
        class_gain = (self.max_delta * torch.sigmoid(self.gain_raw)).view(1, self.nc, 1, 1).to(dtype=base_logits.dtype)
        delta = (class_gain * class_gate * local_support * missing_condition * prior * localization_guard).clamp(0.0, self.max_delta)
        return base_logits + delta, {
            "delta": delta,
            "candidate_logits": candidate_logits,
            "class_gate": class_gate,
            "spatial_prior": spatial_prior,
            "localization_guard": localization_guard,
        }


class CPCRDetect(Detect):
    """Three-scale Detect with detached P2-guided, class-specific bounded P3-logit recovery."""

    def __init__(
        self,
        nc=80,
        max_delta=1.50,
        loc_floor=0.50,
        support_self=0.75,
        miss_power=1.0,
        reduction=4,
        candidate_bias=-2.0,
        gain_init=-2.2,
        use_spatial_prior=True,
        use_class_gate=True,
        use_loc_guard=True,
        detach_guidance=True,
        c_shallow=0,
        ch=(),
    ):
        if not isinstance(ch, (list, tuple)) or len(ch) != 3:
            raise ValueError(f"CPCRDetect requires P3/P4/P5 channels, got {ch}.")
        self.c_shallow = int(c_shallow)
        if self.c_shallow <= 0:
            raise ValueError(f"c_shallow must be positive, got {self.c_shallow}.")
        super().__init__(nc=nc, ch=ch)
        if self.end2end:
            raise NotImplementedError("CPCRDetect supports only the standard one-to-many path.")
        if self.nl != 3:
            raise ValueError(f"CPCRDetect requires exactly three detection levels, got {self.nl}.")
        if any(len(tower) != 3 for tower in self.cv3) or not isinstance(self.cv3[0][-1], nn.Conv2d):
            raise RuntimeError("CPCRDetect expects non-legacy cv3 towers with a final classifier Conv2d.")
        c_hidden = int(self.cv3[0][-1].in_channels)
        self.recovery = ClassPrototypeComplementaryRecovery(
            c_shallow=self.c_shallow,
            c_p3=int(ch[0]),
            c_hidden=c_hidden,
            nc=int(nc),
            reg_max=self.reg_max,
            max_delta=max_delta,
            loc_floor=loc_floor,
            support_self=support_self,
            miss_power=miss_power,
            reduction=reduction,
            candidate_bias=candidate_bias,
            gain_init=gain_init,
            use_spatial_prior=use_spatial_prior,
            use_class_gate=use_class_gate,
            use_loc_guard=use_loc_guard,
            detach_guidance=detach_guidance,
        )
        self.latest_diagnostics = None

    def bias_init(self):
        super().bias_init()
        nn.init.constant_(self.recovery.candidate_bias, self.recovery.candidate_bias_init)

    def forward(self, x):
        if not isinstance(x, (list, tuple)) or len(x) != 4:
            raise ValueError("CPCRDetect expects [P2, P3, P4, P5].")
        shallow, p3, p4, p5 = x
        if shallow.ndim != 4 or shallow.shape[1] != self.c_shallow:
            raise ValueError(f"Expected NCHW P2 guidance with channels={self.c_shallow}, got {tuple(shallow.shape)}.")
        outputs = []
        for index, feature in enumerate((p3, p4, p5)):
            box_logits = self.cv2[index](feature)
            if index == 0:
                base_hidden = self.cv3[index][0](feature)
                base_hidden = self.cv3[index][1](base_hidden)
                base_logits = self.cv3[index][2](base_hidden)
                class_logits, diagnostics = self.recovery(
                    shallow=shallow,
                    p3=p3,
                    base_hidden=base_hidden,
                    base_logits=base_logits,
                    box_logits=box_logits,
                    classifier_conv=self.cv3[index][2],
                )
                if self.training:
                    self.latest_diagnostics = {key: value.detach() for key, value in diagnostics.items()}
            else:
                class_logits = self.cv3[index](feature)
            outputs.append(torch.cat((box_logits, class_logits), dim=1))
        if self.training:
            return outputs
        prediction = self._inference(outputs)
        return prediction if self.export else (prediction, outputs)


class Segment(Detect):
    """YOLO Segment head for segmentation models."""

    def __init__(self, nc=80, nm=32, npr=256, ch=()):
        """Initialize the YOLO model attributes such as the number of masks, prototypes, and the convolution layers."""
        super().__init__(nc, ch)
        self.nm = nm  # number of masks
        self.npr = npr  # number of protos
        self.proto = Proto(ch[0], self.npr, self.nm)  # protos

        c4 = max(ch[0] // 4, self.nm)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nm, 1)) for x in ch)

    def forward(self, x):
        """Return model outputs and mask coefficients if training, otherwise return outputs and mask coefficients."""
        p = self.proto(x[0])  # mask protos
        bs = p.shape[0]  # batch size

        mc = torch.cat([self.cv4[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)  # mask coefficients
        x = Detect.forward(self, x)
        if self.training:
            return x, mc, p
        return (torch.cat([x, mc], 1), p) if self.export else (torch.cat([x[0], mc], 1), (x[1], mc, p))


class OBB(Detect):
    """YOLO OBB detection head for detection with rotation models."""

    def __init__(self, nc=80, ne=1, ch=()):
        """Initialize OBB with number of classes `nc` and layer channels `ch`."""
        super().__init__(nc, ch)
        self.ne = ne  # number of extra parameters

        c4 = max(ch[0] // 4, self.ne)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.ne, 1)) for x in ch)

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        bs = x[0].shape[0]  # batch size
        angle = torch.cat([self.cv4[i](x[i]).view(bs, self.ne, -1) for i in range(self.nl)], 2)  # OBB theta logits
        # NOTE: set `angle` as an attribute so that `decode_bboxes` could use it.
        angle = (angle.sigmoid() - 0.25) * math.pi  # [-pi/4, 3pi/4]
        # angle = angle.sigmoid() * math.pi / 2  # [0, pi/2]
        if not self.training:
            self.angle = angle
        x = Detect.forward(self, x)
        if self.training:
            return x, angle
        return torch.cat([x, angle], 1) if self.export else (torch.cat([x[0], angle], 1), (x[1], angle))

    def decode_bboxes(self, bboxes, anchors):
        """Decode rotated bounding boxes."""
        return dist2rbox(bboxes, self.angle, anchors, dim=1)


class Pose(Detect):
    """YOLO Pose head for keypoints models."""

    def __init__(self, nc=80, kpt_shape=(17, 3), ch=()):
        """Initialize YOLO network with default parameters and Convolutional Layers."""
        super().__init__(nc, ch)
        self.kpt_shape = kpt_shape  # number of keypoints, number of dims (2 for x,y or 3 for x,y,visible)
        self.nk = kpt_shape[0] * kpt_shape[1]  # number of keypoints total

        c4 = max(ch[0] // 4, self.nk)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nk, 1)) for x in ch)

    def forward(self, x):
        """Perform forward pass through YOLO model and return predictions."""
        bs = x[0].shape[0]  # batch size
        kpt = torch.cat([self.cv4[i](x[i]).view(bs, self.nk, -1) for i in range(self.nl)], -1)  # (bs, 17*3, h*w)
        x = Detect.forward(self, x)
        if self.training:
            return x, kpt
        pred_kpt = self.kpts_decode(bs, kpt)
        return torch.cat([x, pred_kpt], 1) if self.export else (torch.cat([x[0], pred_kpt], 1), (x[1], kpt))

    def kpts_decode(self, bs, kpts):
        """Decodes keypoints."""
        ndim = self.kpt_shape[1]
        if self.export:
            if self.format in {
                "tflite",
                "edgetpu",
            }:  # required for TFLite export to avoid 'PLACEHOLDER_FOR_GREATER_OP_CODES' bug
                # Precompute normalization factor to increase numerical stability
                y = kpts.view(bs, *self.kpt_shape, -1)
                grid_h, grid_w = self.shape[2], self.shape[3]
                grid_size = torch.tensor([grid_w, grid_h], device=y.device).reshape(1, 2, 1)
                norm = self.strides / (self.stride[0] * grid_size)
                a = (y[:, :, :2] * 2.0 + (self.anchors - 0.5)) * norm
            else:
                # NCNN fix
                y = kpts.view(bs, *self.kpt_shape, -1)
                a = (y[:, :, :2] * 2.0 + (self.anchors - 0.5)) * self.strides
            if ndim == 3:
                a = torch.cat((a, y[:, :, 2:3].sigmoid()), 2)
            return a.view(bs, self.nk, -1)
        else:
            y = kpts.clone()
            if ndim == 3:
                y[:, 2::3] = y[:, 2::3].sigmoid()  # sigmoid (WARNING: inplace .sigmoid_() Apple MPS bug)
            y[:, 0::ndim] = (y[:, 0::ndim] * 2.0 + (self.anchors[0] - 0.5)) * self.strides
            y[:, 1::ndim] = (y[:, 1::ndim] * 2.0 + (self.anchors[1] - 0.5)) * self.strides
            return y


class Classify(nn.Module):
    """YOLO classification head, i.e. x(b,c1,20,20) to x(b,c2)."""

    export = False  # export mode

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):
        """Initializes YOLO classification head to transform input tensor from (b,c1,20,20) to (b,c2) shape."""
        super().__init__()
        c_ = 1280  # efficientnet_b0 size
        self.conv = Conv(c1, c_, k, s, p, g)
        self.pool = nn.AdaptiveAvgPool2d(1)  # to x(b,c_,1,1)
        self.drop = nn.Dropout(p=0.0, inplace=True)
        self.linear = nn.Linear(c_, c2)  # to x(b,c2)

    def forward(self, x):
        """Performs a forward pass of the YOLO model on input image data."""
        if isinstance(x, list):
            x = torch.cat(x, 1)
        x = self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))
        if self.training:
            return x
        y = x.softmax(1)  # get final output
        return y if self.export else (y, x)


class WorldDetect(Detect):
    """Head for integrating YOLO detection models with semantic understanding from text embeddings."""

    def __init__(self, nc=80, embed=512, with_bn=False, ch=()):
        """Initialize YOLO detection layer with nc classes and layer channels ch."""
        super().__init__(nc, ch)
        c3 = max(ch[0], min(self.nc, 100))
        self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, embed, 1)) for x in ch)
        self.cv4 = nn.ModuleList(BNContrastiveHead(embed) if with_bn else ContrastiveHead() for _ in ch)

    def forward(self, x, text):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv4[i](self.cv3[i](x[i]), text)), 1)
        if self.training:
            return x

        # Inference path
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.nc + self.reg_max * 4, -1) for xi in x], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        if self.export and self.format in {"saved_model", "pb", "tflite", "edgetpu", "tfjs"}:  # avoid TF FlexSplitV ops
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4 :]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

        if self.export and self.format in {"tflite", "edgetpu"}:
            # Precompute normalization factor to increase numerical stability
            # See https://github.com/ultralytics/ultralytics/issues/7371
            grid_h = shape[2]
            grid_w = shape[3]
            grid_size = torch.tensor([grid_w, grid_h, grid_w, grid_h], device=box.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * grid_size)
            dbox = self.decode_bboxes(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2])
        else:
            dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides

        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            # b[-1].bias.data[:] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)


class RTDETRDecoder(nn.Module):
    """
    Real-Time Deformable Transformer Decoder (RTDETRDecoder) module for object detection.

    This decoder module utilizes Transformer architecture along with deformable convolutions to predict bounding boxes
    and class labels for objects in an image. It integrates features from multiple layers and runs through a series of
    Transformer decoder layers to output the final predictions.
    """

    export = False  # export mode

    def __init__(
        self,
        nc=80,
        ch=(512, 1024, 2048),
        hd=256,  # hidden dim
        nq=300,  # num queries
        ndp=4,  # num decoder points
        nh=8,  # num head
        ndl=6,  # num decoder layers
        d_ffn=1024,  # dim of feedforward
        dropout=0.0,
        act=nn.ReLU(),
        eval_idx=-1,
        # Training args
        nd=100,  # num denoising
        label_noise_ratio=0.5,
        box_noise_scale=1.0,
        learnt_init_query=False,
    ):
        """
        Initializes the RTDETRDecoder module with the given parameters.

        Args:
            nc (int): Number of classes. Default is 80.
            ch (tuple): Channels in the backbone feature maps. Default is (512, 1024, 2048).
            hd (int): Dimension of hidden layers. Default is 256.
            nq (int): Number of query points. Default is 300.
            ndp (int): Number of decoder points. Default is 4.
            nh (int): Number of heads in multi-head attention. Default is 8.
            ndl (int): Number of decoder layers. Default is 6.
            d_ffn (int): Dimension of the feed-forward networks. Default is 1024.
            dropout (float): Dropout rate. Default is 0.
            act (nn.Module): Activation function. Default is nn.ReLU.
            eval_idx (int): Evaluation index. Default is -1.
            nd (int): Number of denoising. Default is 100.
            label_noise_ratio (float): Label noise ratio. Default is 0.5.
            box_noise_scale (float): Box noise scale. Default is 1.0.
            learnt_init_query (bool): Whether to learn initial query embeddings. Default is False.
        """
        super().__init__()
        self.hidden_dim = hd
        self.nhead = nh
        self.nl = len(ch)  # num level
        self.nc = nc
        self.num_queries = nq
        self.num_decoder_layers = ndl

        # Backbone feature projection
        self.input_proj = nn.ModuleList(nn.Sequential(nn.Conv2d(x, hd, 1, bias=False), nn.BatchNorm2d(hd)) for x in ch)
        # NOTE: simplified version but it's not consistent with .pt weights.
        # self.input_proj = nn.ModuleList(Conv(x, hd, act=False) for x in ch)

        # Transformer module
        decoder_layer = DeformableTransformerDecoderLayer(hd, nh, d_ffn, dropout, act, self.nl, ndp)
        self.decoder = DeformableTransformerDecoder(hd, decoder_layer, ndl, eval_idx)

        # Denoising part
        self.denoising_class_embed = nn.Embedding(nc, hd)
        self.num_denoising = nd
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale

        # Decoder embedding
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(nq, hd)
        self.query_pos_head = MLP(4, 2 * hd, hd, num_layers=2)

        # Encoder head
        self.enc_output = nn.Sequential(nn.Linear(hd, hd), nn.LayerNorm(hd))
        self.enc_score_head = nn.Linear(hd, nc)
        self.enc_bbox_head = MLP(hd, hd, 4, num_layers=3)

        # Decoder head
        self.dec_score_head = nn.ModuleList([nn.Linear(hd, nc) for _ in range(ndl)])
        self.dec_bbox_head = nn.ModuleList([MLP(hd, hd, 4, num_layers=3) for _ in range(ndl)])

        self._reset_parameters()

    def forward(self, x, batch=None):
        """Runs the forward pass of the module, returning bounding box and classification scores for the input."""
        from ultralytics.models.utils.ops import get_cdn_group

        # Input projection and embedding
        feats, shapes = self._get_encoder_input(x)

        # Prepare denoising training
        dn_embed, dn_bbox, attn_mask, dn_meta = get_cdn_group(
            batch,
            self.nc,
            self.num_queries,
            self.denoising_class_embed.weight,
            self.num_denoising,
            self.label_noise_ratio,
            self.box_noise_scale,
            self.training,
        )

        embed, refer_bbox, enc_bboxes, enc_scores = self._get_decoder_input(feats, shapes, dn_embed, dn_bbox)

        # Decoder
        dec_bboxes, dec_scores = self.decoder(
            embed,
            refer_bbox,
            feats,
            shapes,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            attn_mask=attn_mask,
        )
        x = dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta
        if self.training:
            return x
        # (bs, 300, 4+nc)
        y = torch.cat((dec_bboxes.squeeze(0), dec_scores.squeeze(0).sigmoid()), -1)
        return y if self.export else (y, x)

    def _generate_anchors(self, shapes, grid_size=0.05, dtype=torch.float32, device="cpu", eps=1e-2):
        """Generates anchor bounding boxes for given shapes with specific grid size and validates them."""
        anchors = []
        for i, (h, w) in enumerate(shapes):
            sy = torch.arange(end=h, dtype=dtype, device=device)
            sx = torch.arange(end=w, dtype=dtype, device=device)
            grid_y, grid_x = torch.meshgrid(sy, sx, indexing="ij") if TORCH_1_10 else torch.meshgrid(sy, sx)
            grid_xy = torch.stack([grid_x, grid_y], -1)  # (h, w, 2)

            valid_WH = torch.tensor([w, h], dtype=dtype, device=device)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH  # (1, h, w, 2)
            wh = torch.ones_like(grid_xy, dtype=dtype, device=device) * grid_size * (2.0**i)
            anchors.append(torch.cat([grid_xy, wh], -1).view(-1, h * w, 4))  # (1, h*w, 4)

        anchors = torch.cat(anchors, 1)  # (1, h*w*nl, 4)
        valid_mask = ((anchors > eps) & (anchors < 1 - eps)).all(-1, keepdim=True)  # 1, h*w*nl, 1
        anchors = torch.log(anchors / (1 - anchors))
        anchors = anchors.masked_fill(~valid_mask, float("inf"))
        return anchors, valid_mask

    def _get_encoder_input(self, x):
        """Processes and returns encoder inputs by getting projection features from input and concatenating them."""
        # Get projection features
        x = [self.input_proj[i](feat) for i, feat in enumerate(x)]
        # Get encoder inputs
        feats = []
        shapes = []
        for feat in x:
            h, w = feat.shape[2:]
            # [b, c, h, w] -> [b, h*w, c]
            feats.append(feat.flatten(2).permute(0, 2, 1))
            # [nl, 2]
            shapes.append([h, w])

        # [b, h*w, c]
        feats = torch.cat(feats, 1)
        return feats, shapes

    def _get_decoder_input(self, feats, shapes, dn_embed=None, dn_bbox=None):
        """Generates and prepares the input required for the decoder from the provided features and shapes."""
        bs = feats.shape[0]
        # Prepare input for decoder
        anchors, valid_mask = self._generate_anchors(shapes, dtype=feats.dtype, device=feats.device)
        features = self.enc_output(valid_mask * feats)  # bs, h*w, 256

        enc_outputs_scores = self.enc_score_head(features)  # (bs, h*w, nc)

        # Query selection
        # (bs, num_queries)
        topk_ind = torch.topk(enc_outputs_scores.max(-1).values, self.num_queries, dim=1).indices.view(-1)
        # (bs, num_queries)
        batch_ind = torch.arange(end=bs, dtype=topk_ind.dtype).unsqueeze(-1).repeat(1, self.num_queries).view(-1)

        # (bs, num_queries, 256)
        top_k_features = features[batch_ind, topk_ind].view(bs, self.num_queries, -1)
        # (bs, num_queries, 4)
        top_k_anchors = anchors[:, topk_ind].view(bs, self.num_queries, -1)

        # Dynamic anchors + static content
        refer_bbox = self.enc_bbox_head(top_k_features) + top_k_anchors

        enc_bboxes = refer_bbox.sigmoid()
        if dn_bbox is not None:
            refer_bbox = torch.cat([dn_bbox, refer_bbox], 1)
        enc_scores = enc_outputs_scores[batch_ind, topk_ind].view(bs, self.num_queries, -1)

        embeddings = self.tgt_embed.weight.unsqueeze(0).repeat(bs, 1, 1) if self.learnt_init_query else top_k_features
        if self.training:
            refer_bbox = refer_bbox.detach()
            if not self.learnt_init_query:
                embeddings = embeddings.detach()
        if dn_embed is not None:
            embeddings = torch.cat([dn_embed, embeddings], 1)

        return embeddings, refer_bbox, enc_bboxes, enc_scores

    # TODO
    def _reset_parameters(self):
        """Initializes or resets the parameters of the model's various components with predefined weights and biases."""
        # Class and bbox head init
        bias_cls = bias_init_with_prob(0.01) / 80 * self.nc
        # NOTE: the weight initialization in `linear_init` would cause NaN when training with custom datasets.
        # linear_init(self.enc_score_head)
        constant_(self.enc_score_head.bias, bias_cls)
        constant_(self.enc_bbox_head.layers[-1].weight, 0.0)
        constant_(self.enc_bbox_head.layers[-1].bias, 0.0)
        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            # linear_init(cls_)
            constant_(cls_.bias, bias_cls)
            constant_(reg_.layers[-1].weight, 0.0)
            constant_(reg_.layers[-1].bias, 0.0)

        linear_init(self.enc_output[0])
        xavier_uniform_(self.enc_output[0].weight)
        if self.learnt_init_query:
            xavier_uniform_(self.tgt_embed.weight)
        xavier_uniform_(self.query_pos_head.layers[0].weight)
        xavier_uniform_(self.query_pos_head.layers[1].weight)
        for layer in self.input_proj:
            xavier_uniform_(layer[0].weight)


class v10Detect(Detect):
    """
    v10 Detection head from https://arxiv.org/pdf/2405.14458.

    Args:
        nc (int): Number of classes.
        ch (tuple): Tuple of channel sizes.

    Attributes:
        max_det (int): Maximum number of detections.

    Methods:
        __init__(self, nc=80, ch=()): Initializes the v10Detect object.
        forward(self, x): Performs forward pass of the v10Detect module.
        bias_init(self): Initializes biases of the Detect module.

    """

    end2end = True

    def __init__(self, nc=80, ch=()):
        """Initializes the v10Detect object with the specified number of classes and input channels."""
        super().__init__(nc, ch)
        c3 = max(ch[0], min(self.nc, 100))  # channels
        # Light cls head
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(Conv(x, x, 3, g=x), Conv(x, c3, 1)),
                nn.Sequential(Conv(c3, c3, 3, g=c3), Conv(c3, c3, 1)),
                nn.Conv2d(c3, self.nc, 1),
            )
            for x in ch
        )
        self.one2one_cv3 = copy.deepcopy(self.cv3)
