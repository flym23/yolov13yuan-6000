# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Block modules."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from ultralytics.utils.torch_utils import fuse_conv_and_bn
from .conv import Conv, DSConv, DWConv, GhostConv, LightConv, RepConv, autopad
from .transformer import TransformerBlock

__all__ = (
    "DFL",
    "HGBlock",
    "HGStem",
    "SPP",
    "SPPF",
    "C1",
    "C2",
    "C3",
    "C2f",
    "C2fAttn",
    "ImagePoolingAttn",
    "ContrastiveHead",
    "BNContrastiveHead",
    "C3x",
    "C3TR",
    "C3Ghost",
    "GhostBottleneck",
    "Bottleneck",
    "BottleneckCSP",
    "Proto",
    "RepC3",
    "ResNetLayer",
    "RepNCSPELAN4",
    "ELAN1",
    "ADown",
    "AConv",
    "SPPELAN",
    "CBFuse",
    "CBLinear",
    "C3k2",
    "C2fPSA",
    "C2PSA",
    "RepVGGDW",
    "CIB",
    "C2fCIB",
    "Attention",
    "PSA",
    "SCDown",
    "TorchVision",
    "HyperACE", 
    "DownsampleConv", 
    "FullPAD_Tunnel",
    "DSC3k2",
    "DRSC",
    "DRSCDSC3k2",
    "DAPD",
    "ContourGuidedAdaptiveGeometry",
    "CAGDSC3k2",
    "ShallowEvidenceRouter",
    "CenterPreservedPartialGeometry",
    "SCPGDSC3k2",
    "ConsensusBudgetedEvidenceRouter",
    "CBERSCPGDSC3k2",
    "BCRAUp",
    "MCASUp",
    "ConsensusReliabilityRouter",
    "SymmetricMomentPreservedGeometry",
    "CMRFDSC3k2",
    "ReliabilityFrequencyAlignUp",
    "MicroObjectMomentRefiner",
    "CARMDSC3k2",
    "OrthogonalComplementaryAlignUp",
    "MissingAwareCandidateReactivator",
    "MACRDSC3k2",
)


class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, _, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


class Proto(nn.Module):
    """YOLOv8 mask Proto module for segmentation models."""

    def __init__(self, c1, c_=256, c2=32):
        """
        Initializes the YOLOv8 mask Proto module with specified number of protos and masks.

        Input arguments are ch_in, number of protos, number of masks.
        """
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x):
        """Performs a forward pass through layers using an upsampled input image."""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class HGStem(nn.Module):
    """
    StemBlock of PPHGNetV2 with 5 convolutions and one maxpool2d.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2):
        """Initialize the SPP layer with input/output channels and specified kernel sizes for max pooling."""
        super().__init__()
        self.stem1 = Conv(c1, cm, 3, 2, act=nn.ReLU())
        self.stem2a = Conv(cm, cm // 2, 2, 1, 0, act=nn.ReLU())
        self.stem2b = Conv(cm // 2, cm, 2, 1, 0, act=nn.ReLU())
        self.stem3 = Conv(cm * 2, cm, 3, 2, act=nn.ReLU())
        self.stem4 = Conv(cm, c2, 1, 1, act=nn.ReLU())
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        x = self.stem1(x)
        x = F.pad(x, [0, 1, 0, 1])
        x2 = self.stem2a(x)
        x2 = F.pad(x2, [0, 1, 0, 1])
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class HGBlock(nn.Module):
    """
    HG_Block of PPHGNetV2 with 2 convolutions and LightConv.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2, k=3, n=6, lightconv=False, shortcut=False, act=nn.ReLU()):
        """Initializes a CSP Bottleneck with 1 convolution using specified input and output channels."""
        super().__init__()
        block = LightConv if lightconv else Conv
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)  # squeeze conv
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)  # excitation conv
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(torch.cat(y, 1)))
        return y + x if self.add else y


class SPP(nn.Module):
    """Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729."""

    def __init__(self, c1, c2, k=(5, 9, 13)):
        """Initialize the SPP layer with input/output channels and pooling kernel sizes."""
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        """Forward pass of the SPP layer, performing spatial pyramid pooling."""
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):
        """
        Initializes the SPPF layer with given input/output channels and kernel size.

        This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))


class C1(nn.Module):
    """CSP Bottleneck with 1 convolution."""

    def __init__(self, c1, c2, n=1):
        """Initializes the CSP Bottleneck with configurations for 1 convolution with arguments ch_in, ch_out, number."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))

    def forward(self, x):
        """Applies cross-convolutions to input in the C3 module."""
        y = self.cv1(x)
        return self.m(y) + y


class C2(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes a CSP Bottleneck with 2 convolutions and optional shortcut connection."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)  # optional act=FReLU(c2)
        # self.attention = ChannelAttention(2 * self.c)  # or SpatialAttention()
        self.m = nn.Sequential(*(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(a), b), 1))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3x(C3):
    """C3 module with cross-convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3TR instance and set default parameters."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1) for _ in range(n)))


class RepC3(nn.Module):
    """Rep C3."""

    def __init__(self, c1, c2, n=3, e=1.0):
        """Initialize CSP Bottleneck with a single convolution using input channels, output channels, and number."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])
        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()

    def forward(self, x):
        """Forward pass of RT-DETR neck layer."""
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))


class C3TR(C3):
    """C3 module with TransformerBlock()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3Ghost module with GhostBottleneck()."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3Ghost(C3):
    """C3 module with GhostBottleneck()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize 'SPP' module with various pooling sizes for spatial pyramid pooling."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class GhostBottleneck(nn.Module):
    """Ghost Bottleneck https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=3, s=1):
        """Initializes GhostBottleneck module with arguments ch_in, ch_out, kernel, stride."""
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False),  # pw-linear
        )
        self.shortcut = (
            nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()
        )

    def forward(self, x):
        """Applies skip connection and concatenation to input tensor."""
        return self.conv(x) + self.shortcut(x)


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    """CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes the CSP Bottleneck given arguments for ch_in, ch_out, number, shortcut, groups, expansion."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        """Applies a CSP bottleneck with 3 convolutions."""
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


class ResNetBlock(nn.Module):
    """ResNet block with standard convolution layers."""

    def __init__(self, c1, c2, s=1, e=4):
        """Initialize convolution with given parameters."""
        super().__init__()
        c3 = e * c2
        self.cv1 = Conv(c1, c2, k=1, s=1, act=True)
        self.cv2 = Conv(c2, c2, k=3, s=s, p=1, act=True)
        self.cv3 = Conv(c2, c3, k=1, act=False)
        self.shortcut = nn.Sequential(Conv(c1, c3, k=1, s=s, act=False)) if s != 1 or c1 != c3 else nn.Identity()

    def forward(self, x):
        """Forward pass through the ResNet block."""
        return F.relu(self.cv3(self.cv2(self.cv1(x))) + self.shortcut(x))


class ResNetLayer(nn.Module):
    """ResNet layer with multiple ResNet blocks."""

    def __init__(self, c1, c2, s=1, is_first=False, n=1, e=4):
        """Initializes the ResNetLayer given arguments."""
        super().__init__()
        self.is_first = is_first

        if self.is_first:
            self.layer = nn.Sequential(
                Conv(c1, c2, k=7, s=2, p=3, act=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        else:
            blocks = [ResNetBlock(c1, c2, s, e=e)]
            blocks.extend([ResNetBlock(e * c2, c2, 1, e=e) for _ in range(n - 1)])
            self.layer = nn.Sequential(*blocks)

    def forward(self, x):
        """Forward pass through the ResNet layer."""
        return self.layer(x)


class MaxSigmoidAttnBlock(nn.Module):
    """Max Sigmoid attention block."""

    def __init__(self, c1, c2, nh=1, ec=128, gc=512, scale=False):
        """Initializes MaxSigmoidAttnBlock with specified arguments."""
        super().__init__()
        self.nh = nh
        self.hc = c2 // nh
        self.ec = Conv(c1, ec, k=1, act=False) if c1 != ec else None
        self.gl = nn.Linear(gc, ec)
        self.bias = nn.Parameter(torch.zeros(nh))
        self.proj_conv = Conv(c1, c2, k=3, s=1, act=False)
        self.scale = nn.Parameter(torch.ones(1, nh, 1, 1)) if scale else 1.0

    def forward(self, x, guide):
        """Forward process."""
        bs, _, h, w = x.shape

        guide = self.gl(guide)
        guide = guide.view(bs, -1, self.nh, self.hc)
        embed = self.ec(x) if self.ec is not None else x
        embed = embed.view(bs, self.nh, self.hc, h, w)

        aw = torch.einsum("bmchw,bnmc->bmhwn", embed, guide)
        aw = aw.max(dim=-1)[0]
        aw = aw / (self.hc**0.5)
        aw = aw + self.bias[None, :, None, None]
        aw = aw.sigmoid() * self.scale

        x = self.proj_conv(x)
        x = x.view(bs, self.nh, -1, h, w)
        x = x * aw.unsqueeze(2)
        return x.view(bs, -1, h, w)


class C2fAttn(nn.Module):
    """C2f module with an additional attn module."""

    def __init__(self, c1, c2, n=1, ec=128, nh=1, gc=512, shortcut=False, g=1, e=0.5):
        """Initializes C2f module with attention mechanism for enhanced feature extraction and processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((3 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.attn = MaxSigmoidAttnBlock(self.c, self.c, gc=gc, ec=ec, nh=nh)

    def forward(self, x, guide):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x, guide):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))


class ImagePoolingAttn(nn.Module):
    """ImagePoolingAttn: Enhance the text embeddings with image-aware information."""

    def __init__(self, ec=256, ch=(), ct=512, nh=8, k=3, scale=False):
        """Initializes ImagePoolingAttn with specified arguments."""
        super().__init__()

        nf = len(ch)
        self.query = nn.Sequential(nn.LayerNorm(ct), nn.Linear(ct, ec))
        self.key = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.value = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.proj = nn.Linear(ec, ct)
        self.scale = nn.Parameter(torch.tensor([0.0]), requires_grad=True) if scale else 1.0
        self.projections = nn.ModuleList([nn.Conv2d(in_channels, ec, kernel_size=1) for in_channels in ch])
        self.im_pools = nn.ModuleList([nn.AdaptiveMaxPool2d((k, k)) for _ in range(nf)])
        self.ec = ec
        self.nh = nh
        self.nf = nf
        self.hc = ec // nh
        self.k = k

    def forward(self, x, text):
        """Executes attention mechanism on input tensor x and guide tensor."""
        bs = x[0].shape[0]
        assert len(x) == self.nf
        num_patches = self.k**2
        x = [pool(proj(x)).view(bs, -1, num_patches) for (x, proj, pool) in zip(x, self.projections, self.im_pools)]
        x = torch.cat(x, dim=-1).transpose(1, 2)
        q = self.query(text)
        k = self.key(x)
        v = self.value(x)

        # q = q.reshape(1, text.shape[1], self.nh, self.hc).repeat(bs, 1, 1, 1)
        q = q.reshape(bs, -1, self.nh, self.hc)
        k = k.reshape(bs, -1, self.nh, self.hc)
        v = v.reshape(bs, -1, self.nh, self.hc)

        aw = torch.einsum("bnmc,bkmc->bmnk", q, k)
        aw = aw / (self.hc**0.5)
        aw = F.softmax(aw, dim=-1)

        x = torch.einsum("bmnk,bkmc->bnmc", aw, v)
        x = self.proj(x.reshape(bs, -1, self.ec))
        return x * self.scale + text


class ContrastiveHead(nn.Module):
    """Implements contrastive learning head for region-text similarity in vision-language models."""

    def __init__(self):
        """Initializes ContrastiveHead with specified region-text similarity parameters."""
        super().__init__()
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.tensor(1 / 0.07).log())

    def forward(self, x, w):
        """Forward function of contrastive learning."""
        x = F.normalize(x, dim=1, p=2)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class BNContrastiveHead(nn.Module):
    """
    Batch Norm Contrastive Head for YOLO-World using batch norm instead of l2-normalization.

    Args:
        embed_dims (int): Embed dimensions of text and image features.
    """

    def __init__(self, embed_dims: int):
        """Initialize ContrastiveHead with region-text similarity parameters."""
        super().__init__()
        self.norm = nn.BatchNorm2d(embed_dims)
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        # use -1.0 is more stable
        self.logit_scale = nn.Parameter(-1.0 * torch.ones([]))

    def forward(self, x, w):
        """Forward function of contrastive learning."""
        x = self.norm(x)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class RepBottleneck(Bottleneck):
    """Rep bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a RepBottleneck module with customizable in/out channels, shortcuts, groups and expansion."""
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = RepConv(c1, c_, k[0], 1)


class RepCSP(C3):
    """Repeatable Cross Stage Partial Network (RepCSP) module for efficient feature extraction."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes RepCSP layer with given channels, repetitions, shortcut, groups and expansion ratio."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))


class RepNCSPELAN4(nn.Module):
    """CSP-ELAN."""

    def __init__(self, c1, c2, c3, c4, n=1):
        """Initializes CSP-ELAN layer with specified channel sizes, repetitions, and convolutions."""
        super().__init__()
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.Sequential(RepCSP(c3 // 2, c4, n), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(RepCSP(c4, c4, n), Conv(c4, c4, 3, 1))
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)

    def forward(self, x):
        """Forward pass through RepNCSPELAN4 layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


class ELAN1(RepNCSPELAN4):
    """ELAN1 module with 4 convolutions."""

    def __init__(self, c1, c2, c3, c4):
        """Initializes ELAN1 layer with specified channel sizes."""
        super().__init__(c1, c2, c3, c4)
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = Conv(c3 // 2, c4, 3, 1)
        self.cv3 = Conv(c4, c4, 3, 1)
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)


class AConv(nn.Module):
    """AConv."""

    def __init__(self, c1, c2):
        """Initializes AConv module with convolution layers."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 3, 2, 1)

    def forward(self, x):
        """Forward pass through AConv layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        return self.cv1(x)


class ADown(nn.Module):
    """ADown."""

    def __init__(self, c1, c2):
        """Initializes ADown module with convolution layers to downsample input from channels c1 to c2."""
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)

    def forward(self, x):
        """Forward pass through ADown layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        x2 = self.cv2(x2)
        return torch.cat((x1, x2), 1)


class SPPELAN(nn.Module):
    """SPP-ELAN."""

    def __init__(self, c1, c2, c3, k=5):
        """Initializes SPP-ELAN block with convolution and max pooling layers for spatial pyramid pooling."""
        super().__init__()
        self.c = c3
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv3 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv4 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv5 = Conv(4 * c3, c2, 1, 1)

    def forward(self, x):
        """Forward pass through SPPELAN layer."""
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3, self.cv4])
        return self.cv5(torch.cat(y, 1))


class CBLinear(nn.Module):
    """CBLinear."""

    def __init__(self, c1, c2s, k=1, s=1, p=None, g=1):
        """Initializes the CBLinear module, passing inputs unchanged."""
        super().__init__()
        self.c2s = c2s
        self.conv = nn.Conv2d(c1, sum(c2s), k, s, autopad(k, p), groups=g, bias=True)

    def forward(self, x):
        """Forward pass through CBLinear layer."""
        return self.conv(x).split(self.c2s, dim=1)


class CBFuse(nn.Module):
    """CBFuse."""

    def __init__(self, idx):
        """Initializes CBFuse module with layer index for selective feature fusion."""
        super().__init__()
        self.idx = idx

    def forward(self, xs):
        """Forward pass through CBFuse layer."""
        target_size = xs[-1].shape[2:]
        res = [F.interpolate(x[self.idx[i]], size=target_size, mode="nearest") for i, x in enumerate(xs[:-1])]
        return torch.sum(torch.stack(res + xs[-1:]), dim=0)


class C3f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv((2 + n) * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(c_, c_, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = [self.cv2(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv3(torch.cat(y, 1))


class C3k2(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )


class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class RepVGGDW(torch.nn.Module):
    """RepVGGDW is a class that represents a depth wise separable convolutional block in RepVGG architecture."""

    def __init__(self, ed) -> None:
        """Initializes RepVGGDW with depthwise separable convolutional layers for efficient processing."""
        super().__init__()
        self.conv = Conv(ed, ed, 7, 1, 3, g=ed, act=False)
        self.conv1 = Conv(ed, ed, 3, 1, 1, g=ed, act=False)
        self.dim = ed
        self.act = nn.SiLU()

    def forward(self, x):
        """
        Performs a forward pass of the RepVGGDW block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying the depth wise separable convolution.
        """
        return self.act(self.conv(x) + self.conv1(x))

    def forward_fuse(self, x):
        """
        Performs a forward pass of the RepVGGDW block without fusing the convolutions.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying the depth wise separable convolution.
        """
        return self.act(self.conv(x))

    @torch.no_grad()
    def fuse(self):
        """
        Fuses the convolutional layers in the RepVGGDW block.

        This method fuses the convolutional layers and updates the weights and biases accordingly.
        """
        conv = fuse_conv_and_bn(self.conv.conv, self.conv.bn)
        conv1 = fuse_conv_and_bn(self.conv1.conv, self.conv1.bn)

        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias

        conv1_w = torch.nn.functional.pad(conv1_w, [2, 2, 2, 2])

        final_conv_w = conv_w + conv1_w
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)

        self.conv = conv
        del self.conv1


class CIB(nn.Module):
    """
    Conditional Identity Block (CIB) module.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        shortcut (bool, optional): Whether to add a shortcut connection. Defaults to True.
        e (float, optional): Scaling factor for the hidden channels. Defaults to 0.5.
        lk (bool, optional): Whether to use RepVGGDW for the third convolutional layer. Defaults to False.
    """

    def __init__(self, c1, c2, shortcut=True, e=0.5, lk=False):
        """Initializes the custom model with optional shortcut, scaling factor, and RepVGGDW layer."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = nn.Sequential(
            Conv(c1, c1, 3, g=c1),
            Conv(c1, 2 * c_, 1),
            RepVGGDW(2 * c_) if lk else Conv(2 * c_, 2 * c_, 3, g=2 * c_),
            Conv(2 * c_, c2, 1),
            Conv(c2, c2, 3, g=c2),
        )

        self.add = shortcut and c1 == c2

    def forward(self, x):
        """
        Forward pass of the CIB module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return x + self.cv1(x) if self.add else self.cv1(x)


class C2fCIB(C2f):
    """
    C2fCIB class represents a convolutional block with C2f and CIB modules.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        n (int, optional): Number of CIB modules to stack. Defaults to 1.
        shortcut (bool, optional): Whether to use shortcut connection. Defaults to False.
        lk (bool, optional): Whether to use local key connection. Defaults to False.
        g (int, optional): Number of groups for grouped convolution. Defaults to 1.
        e (float, optional): Expansion ratio for CIB modules. Defaults to 0.5.
    """

    def __init__(self, c1, c2, n=1, shortcut=False, lk=False, g=1, e=0.5):
        """Initializes the module with specified parameters for channel, shortcut, local key, groups, and expansion."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(CIB(self.c, self.c, shortcut, e=1.0, lk=lk) for _ in range(n))


class Attention(nn.Module):
    """
    Attention module that performs self-attention on the input tensor.

    Args:
        dim (int): The input tensor dimension.
        num_heads (int): The number of attention heads.
        attn_ratio (float): The ratio of the attention key dimension to the head dimension.

    Attributes:
        num_heads (int): The number of attention heads.
        head_dim (int): The dimension of each attention head.
        key_dim (int): The dimension of the attention key.
        scale (float): The scaling factor for the attention scores.
        qkv (Conv): Convolutional layer for computing the query, key, and value.
        proj (Conv): Convolutional layer for projecting the attended values.
        pe (Conv): Convolutional layer for positional encoding.
    """

    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        """Initializes multi-head attention module with query, key, and value convolutions and positional encoding."""
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x):
        """
        Forward pass of the Attention module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            (torch.Tensor): The output tensor after self-attention.
        """
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x


class PSABlock(nn.Module):
    """
    PSABlock class implementing a Position-Sensitive Attention block for neural networks.

    This class encapsulates the functionality for applying multi-head attention and feed-forward neural network layers
    with optional shortcut connections.

    Attributes:
        attn (Attention): Multi-head attention module.
        ffn (nn.Sequential): Feed-forward neural network module.
        add (bool): Flag indicating whether to add shortcut connections.

    Methods:
        forward: Performs a forward pass through the PSABlock, applying attention and feed-forward layers.

    Examples:
        Create a PSABlock and perform a forward pass
        >>> psablock = PSABlock(c=128, attn_ratio=0.5, num_heads=4, shortcut=True)
        >>> input_tensor = torch.randn(1, 128, 32, 32)
        >>> output_tensor = psablock(input_tensor)
    """

    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        """Initializes the PSABlock with attention and feed-forward layers for enhanced feature extraction."""
        super().__init__()

        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x):
        """Executes a forward pass through PSABlock, applying attention and feed-forward layers to the input tensor."""
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class PSA(nn.Module):
    """
    PSA class for implementing Position-Sensitive Attention in neural networks.

    This class encapsulates the functionality for applying position-sensitive attention and feed-forward networks to
    input tensors, enhancing feature extraction and processing capabilities.

    Attributes:
        c (int): Number of hidden channels after applying the initial convolution.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        attn (Attention): Attention module for position-sensitive attention.
        ffn (nn.Sequential): Feed-forward network for further processing.

    Methods:
        forward: Applies position-sensitive attention and feed-forward network to the input tensor.

    Examples:
        Create a PSA module and apply it to an input tensor
        >>> psa = PSA(c1=128, c2=128, e=0.5)
        >>> input_tensor = torch.randn(1, 128, 64, 64)
        >>> output_tensor = psa.forward(input_tensor)
    """

    def __init__(self, c1, c2, e=0.5):
        """Initializes the PSA module with input/output channels and attention mechanism for feature extraction."""
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.attn = Attention(self.c, attn_ratio=0.5, num_heads=self.c // 64)
        self.ffn = nn.Sequential(Conv(self.c, self.c * 2, 1), Conv(self.c * 2, self.c, 1, act=False))

    def forward(self, x):
        """Executes forward pass in PSA module, applying attention and feed-forward layers to the input tensor."""
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = b + self.attn(b)
        b = b + self.ffn(b)
        return self.cv2(torch.cat((a, b), 1))


class C2PSA(nn.Module):
    """
    C2PSA module with attention mechanism for enhanced feature extraction and processing.

    This module implements a convolutional block with attention mechanisms to enhance feature extraction and processing
    capabilities. It includes a series of PSABlock modules for self-attention and feed-forward operations.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.Sequential): Sequential container of PSABlock modules for attention and feed-forward operations.

    Methods:
        forward: Performs a forward pass through the C2PSA module, applying attention and feed-forward operations.

    Notes:
        This module essentially is the same as PSA module, but refactored to allow stacking more PSABlock modules.

    Examples:
        >>> c2psa = C2PSA(c1=256, c2=256, n=3, e=0.5)
        >>> input_tensor = torch.randn(1, 256, 64, 64)
        >>> output_tensor = c2psa(input_tensor)
    """

    def __init__(self, c1, c2, n=1, e=0.5):
        """Initializes the C2PSA module with specified input/output channels, number of layers, and expansion ratio."""
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x):
        """Processes the input tensor 'x' through a series of PSA blocks and returns the transformed tensor."""
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))


class C2fPSA(C2f):
    """
    C2fPSA module with enhanced feature extraction using PSA blocks.

    This class extends the C2f module by incorporating PSA blocks for improved attention mechanisms and feature extraction.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.ModuleList): List of PSA blocks for feature extraction.

    Methods:
        forward: Performs a forward pass through the C2fPSA module.
        forward_split: Performs a forward pass using split() instead of chunk().

    Examples:
        >>> import torch
        >>> from ultralytics.models.common import C2fPSA
        >>> model = C2fPSA(c1=64, c2=64, n=3, e=0.5)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)
    """

    def __init__(self, c1, c2, n=1, e=0.5):
        """Initializes the C2fPSA module, a variant of C2f with PSA blocks for enhanced feature extraction."""
        assert c1 == c2
        super().__init__(c1, c2, n=n, e=e)
        self.m = nn.ModuleList(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n))


class SCDown(nn.Module):
    """
    SCDown module for downsampling with separable convolutions.

    This module performs downsampling using a combination of pointwise and depthwise convolutions, which helps in
    efficiently reducing the spatial dimensions of the input tensor while maintaining the channel information.

    Attributes:
        cv1 (Conv): Pointwise convolution layer that reduces the number of channels.
        cv2 (Conv): Depthwise convolution layer that performs spatial downsampling.

    Methods:
        forward: Applies the SCDown module to the input tensor.

    Examples:
        >>> import torch
        >>> from ultralytics import SCDown
        >>> model = SCDown(c1=64, c2=128, k=3, s=2)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> y = model(x)
        >>> print(y.shape)
        torch.Size([1, 128, 64, 64])
    """

    def __init__(self, c1, c2, k, s):
        """Initializes the SCDown module with specified input/output channels, kernel size, and stride."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c2, c2, k=k, s=s, g=c2, act=False)

    def forward(self, x):
        """Applies convolution and downsampling to the input tensor in the SCDown module."""
        return self.cv2(self.cv1(x))


class TorchVision(nn.Module):
    """
    TorchVision module to allow loading any torchvision model.

    This class provides a way to load a model from the torchvision library, optionally load pre-trained weights, and customize the model by truncating or unwrapping layers.

    Attributes:
        m (nn.Module): The loaded torchvision model, possibly truncated and unwrapped.

    Args:
        c1 (int): Input channels.
        c2 (): Output channels.
        model (str): Name of the torchvision model to load.
        weights (str, optional): Pre-trained weights to load. Default is "DEFAULT".
        unwrap (bool, optional): If True, unwraps the model to a sequential containing all but the last `truncate` layers. Default is True.
        truncate (int, optional): Number of layers to truncate from the end if `unwrap` is True. Default is 2.
        split (bool, optional): Returns output from intermediate child modules as list. Default is False.
    """

    def __init__(self, c1, c2, model, weights="DEFAULT", unwrap=True, truncate=2, split=False):
        """Load the model and weights from torchvision."""
        import torchvision  # scope for faster 'import ultralytics'

        super().__init__()
        if hasattr(torchvision.models, "get_model"):
            self.m = torchvision.models.get_model(model, weights=weights)
        else:
            self.m = torchvision.models.__dict__[model](pretrained=bool(weights))
        if unwrap:
            layers = list(self.m.children())[:-truncate]
            if isinstance(layers[0], nn.Sequential):  # Second-level for some models like EfficientNet, Swin
                layers = [*list(layers[0].children()), *layers[1:]]
            self.m = nn.Sequential(*layers)
            self.split = split
        else:
            self.split = False
            self.m.head = self.m.heads = nn.Identity()

    def forward(self, x):
        """Forward pass through the model."""
        if self.split:
            y = [x]
            y.extend(m(y[-1]) for m in self.m)
        else:
            y = self.m(x)
        return y

import logging
logger = logging.getLogger(__name__)

USE_FLASH_ATTN = False
try:
    import torch
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:  # Ampere or newer
        from flash_attn.flash_attn_interface import flash_attn_func
        USE_FLASH_ATTN = True
    else:
        from torch.nn.functional import scaled_dot_product_attention as sdpa
        logger.warning("FlashAttention is not available on this device. Using scaled_dot_product_attention instead.")
except Exception:
    from torch.nn.functional import scaled_dot_product_attention as sdpa
    logger.warning("FlashAttention is not available on this device. Using scaled_dot_product_attention instead.")

class AAttn(nn.Module):
    """
    Area-attention module with the requirement of flash attention.

    Attributes:
        dim (int): Number of hidden channels;
        num_heads (int): Number of heads into which the attention mechanism is divided;
        area (int, optional): Number of areas the feature map is divided. Defaults to 1.

    Methods:
        forward: Performs a forward process of input tensor and outputs a tensor after the execution of the area attention mechanism.

    Examples:
        >>> import torch
        >>> from ultralytics.nn.modules import AAttn
        >>> model = AAttn(dim=64, num_heads=2, area=4)
        >>> x = torch.randn(2, 64, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)
    
    Notes: 
        recommend that dim//num_heads be a multiple of 32 or 64.

    """

    def __init__(self, dim, num_heads, area=1):
        """Initializes the area-attention module, a simple yet efficient attention module for YOLO."""
        super().__init__()
        self.area = area

        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        all_head_dim = head_dim * self.num_heads

        self.qk = Conv(dim, all_head_dim * 2, 1, act=False)
        self.v = Conv(dim, all_head_dim, 1, act=False)
        self.proj = Conv(all_head_dim, dim, 1, act=False)

        self.pe = Conv(all_head_dim, dim, 5, 1, 2, g=dim, act=False)


    def forward(self, x):
        """Processes the input tensor 'x' through the area-attention"""
        B, C, H, W = x.shape
        N = H * W

        qk = self.qk(x).flatten(2).transpose(1, 2)
        v = self.v(x)
        pp = self.pe(v)
        v = v.flatten(2).transpose(1, 2)

        if self.area > 1:
            qk = qk.reshape(B * self.area, N // self.area, C * 2)
            v = v.reshape(B * self.area, N // self.area, C)
            B, N, _ = qk.shape
        q, k = qk.split([C, C], dim=2)

        if x.is_cuda and USE_FLASH_ATTN:
            q = q.view(B, N, self.num_heads, self.head_dim)
            k = k.view(B, N, self.num_heads, self.head_dim)
            v = v.view(B, N, self.num_heads, self.head_dim)

            x = flash_attn_func(
                q.contiguous().half(),
                k.contiguous().half(),
                v.contiguous().half()
            ).to(q.dtype)
        else:
            q = q.transpose(1, 2).view(B, self.num_heads, self.head_dim, N)
            k = k.transpose(1, 2).view(B, self.num_heads, self.head_dim, N)
            v = v.transpose(1, 2).view(B, self.num_heads, self.head_dim, N)

            attn = (q.transpose(-2, -1) @ k) * (self.head_dim ** -0.5)
            max_attn = attn.max(dim=-1, keepdim=True).values
            exp_attn = torch.exp(attn - max_attn)
            attn = exp_attn / exp_attn.sum(dim=-1, keepdim=True)
            x = (v @ attn.transpose(-2, -1))

            x = x.permute(0, 3, 1, 2)

        if self.area > 1:
            x = x.reshape(B // self.area, N * self.area, C)
            B, N, _ = x.shape
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)

        return self.proj(x + pp)
    

class ABlock(nn.Module):
    """
    ABlock class implementing a Area-Attention block with effective feature extraction.

    This class encapsulates the functionality for applying multi-head attention with feature map are dividing into areas
    and feed-forward neural network layers.

    Attributes:
        dim (int): Number of hidden channels;
        num_heads (int): Number of heads into which the attention mechanism is divided;
        mlp_ratio (float, optional): MLP expansion ratio (or MLP hidden dimension ratio). Defaults to 1.2;
        area (int, optional): Number of areas the feature map is divided.  Defaults to 1.

    Methods:
        forward: Performs a forward pass through the ABlock, applying area-attention and feed-forward layers.

    Examples:
        Create a ABlock and perform a forward pass
        >>> model = ABlock(dim=64, num_heads=2, mlp_ratio=1.2, area=4)
        >>> x = torch.randn(2, 64, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)
    
    Notes: 
        recommend that dim//num_heads be a multiple of 32 or 64.
    """

    def __init__(self, dim, num_heads, mlp_ratio=1.2, area=1):
        """Initializes the ABlock with area-attention and feed-forward layers for faster feature extraction."""
        super().__init__()

        self.attn = AAttn(dim, num_heads=num_heads, area=area)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(Conv(dim, mlp_hidden_dim, 1), Conv(mlp_hidden_dim, dim, 1, act=False))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Initialize weights using a truncated normal distribution."""
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Executes a forward pass through ABlock, applying area-attention and feed-forward layers to the input tensor."""
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class A2C2f(nn.Module):  
    """
    A2C2f module with residual enhanced feature extraction using ABlock blocks with area-attention. Also known as R-ELAN

    This class extends the C2f module by incorporating ABlock blocks for fast attention mechanisms and feature extraction.

    Attributes:
        c1 (int): Number of input channels;
        c2 (int): Number of output channels;
        n (int, optional): Number of 2xABlock modules to stack. Defaults to 1;
        a2 (bool, optional): Whether use area-attention. Defaults to True;
        area (int, optional): Number of areas the feature map is divided. Defaults to 1;
        residual (bool, optional): Whether use the residual (with layer scale). Defaults to False;
        mlp_ratio (float, optional): MLP expansion ratio (or MLP hidden dimension ratio). Defaults to 1.2;
        e (float, optional): Expansion ratio for R-ELAN modules. Defaults to 0.5;
        g (int, optional): Number of groups for grouped convolution. Defaults to 1;
        shortcut (bool, optional): Whether to use shortcut connection. Defaults to True;

    Methods:
        forward: Performs a forward pass through the A2C2f module.

    Examples:
        >>> import torch
        >>> from ultralytics.nn.modules import A2C2f
        >>> model = A2C2f(c1=64, c2=64, n=2, a2=True, area=4, residual=True, e=0.5)
        >>> x = torch.randn(2, 64, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)
    """

    def __init__(self, c1, c2, n=1, a2=True, area=1, residual=False, mlp_ratio=2.0, e=0.5, g=1, shortcut=True):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        assert c_ % 32 == 0, "Dimension of ABlock be a multiple of 32."

        # num_heads = c_ // 64 if c_ // 64 >= 2 else c_ // 32
        num_heads = c_ // 32

        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv((1 + n) * c_, c2, 1)  # optional act=FReLU(c2)

        init_values = 0.01  # or smaller
        self.gamma = nn.Parameter(init_values * torch.ones((c2)), requires_grad=True) if a2 and residual else None

        self.m = nn.ModuleList(
            nn.Sequential(*(ABlock(c_, num_heads, mlp_ratio, area) for _ in range(2))) if a2 else C3k(c_, c_, 2, shortcut, g) for _ in range(n)
        )

    def forward(self, x):
        """Forward pass through R-ELAN layer."""
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        if self.gamma is not None:
            return x + self.gamma.view(1, -1, 1, 1) * self.cv2(torch.cat(y, 1))
        return self.cv2(torch.cat(y, 1))

class DSBottleneck(nn.Module):
    """
    An improved bottleneck block using depthwise separable convolutions (DSConv).

    This class implements a lightweight bottleneck module that replaces standard convolutions with depthwise
    separable convolutions to reduce parameters and computational cost. 

    Attributes:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        shortcut (bool, optional): Whether to use a residual shortcut connection. The connection is only added if c1 == c2. Defaults to True.
        e (float, optional): Expansion ratio for the intermediate channels. Defaults to 0.5.
        k1 (int, optional): Kernel size for the first DSConv layer. Defaults to 3.
        k2 (int, optional): Kernel size for the second DSConv layer. Defaults to 5.
        d2 (int, optional): Dilation for the second DSConv layer. Defaults to 1.

    Methods:
        forward: Performs a forward pass through the DSBottleneck module.

    Examples:
        >>> import torch
        >>> model = DSBottleneck(c1=64, c2=64, shortcut=True)
        >>> x = torch.randn(2, 64, 32, 32)
        >>> output = model(x)
        >>> print(output.shape)
        torch.Size([2, 64, 32, 32])
    """
    def __init__(self, c1, c2, shortcut=True, e=0.5, k1=3, k2=5, d2=1):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = DSConv(c1, c_, k1, s=1, p=None, d=1)   
        self.cv2 = DSConv(c_, c2, k2, s=1, p=None, d=d2)  
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class DSC3k(C3):
    """
    An improved C3k module using DSBottleneck blocks for lightweight feature extraction.

    This class extends the C3 module by replacing its standard bottleneck blocks with DSBottleneck blocks,
    which use depthwise separable convolutions.

    Attributes:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        n (int, optional): Number of DSBottleneck blocks to stack. Defaults to 1.
        shortcut (bool, optional): Whether to use shortcut connections within the DSBottlenecks. Defaults to True.
        g (int, optional): Number of groups for grouped convolution (passed to parent C3). Defaults to 1.
        e (float, optional): Expansion ratio for the C3 module's hidden channels. Defaults to 0.5.
        k1 (int, optional): Kernel size for the first DSConv in each DSBottleneck. Defaults to 3.
        k2 (int, optional): Kernel size for the second DSConv in each DSBottleneck. Defaults to 5.
        d2 (int, optional): Dilation for the second DSConv in each DSBottleneck. Defaults to 1.

    Methods:
        forward: Performs a forward pass through the DSC3k module (inherited from C3).

    Examples:
        >>> import torch
        >>> model = DSC3k(c1=128, c2=128, n=2, k1=3, k2=7)
        >>> x = torch.randn(2, 128, 64, 64)
        >>> output = model(x)
        >>> print(output.shape)
        torch.Size([2, 128, 64, 64])
    """
    def __init__(
        self,
        c1,                
        c2,                 
        n=1,                
        shortcut=True,      
        g=1,                 
        e=0.5,              
        k1=3,               
        k2=5,               
        d2=1                 
    ):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  

        self.m = nn.Sequential(
            *(
                DSBottleneck(
                    c_, c_,
                    shortcut=shortcut,
                    e=1.0,
                    k1=k1,
                    k2=k2,
                    d2=d2
                )
                for _ in range(n)
            )
        )

class DSC3k2(C2f):
    """
    An improved C3k2 module that uses lightweight depthwise separable convolution blocks.

    This class redesigns C3k2 module, replacing its internal processing blocks with either DSBottleneck
    or DSC3k modules.

    Attributes:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        n (int, optional): Number of internal processing blocks to stack. Defaults to 1.
        dsc3k (bool, optional): If True, use DSC3k as the internal block. If False, use DSBottleneck. Defaults to False.
        e (float, optional): Expansion ratio for the C2f module's hidden channels. Defaults to 0.5.
        g (int, optional): Number of groups for grouped convolution (passed to parent C2f). Defaults to 1.
        shortcut (bool, optional): Whether to use shortcut connections in the internal blocks. Defaults to True.
        k1 (int, optional): Kernel size for the first DSConv in internal blocks. Defaults to 3.
        k2 (int, optional): Kernel size for the second DSConv in internal blocks. Defaults to 7.
        d2 (int, optional): Dilation for the second DSConv in internal blocks. Defaults to 1.

    Methods:
        forward: Performs a forward pass through the DSC3k2 module (inherited from C2f).

    Examples:
        >>> import torch
        >>> # Using DSBottleneck as internal block
        >>> model1 = DSC3k2(c1=64, c2=64, n=2, dsc3k=False)
        >>> x = torch.randn(2, 64, 128, 128)
        >>> output1 = model1(x)
        >>> print(f"With DSBottleneck: {output1.shape}")
        With DSBottleneck: torch.Size([2, 64, 128, 128])
        >>> # Using DSC3k as internal block
        >>> model2 = DSC3k2(c1=64, c2=64, n=1, dsc3k=True)
        >>> output2 = model2(x)
        >>> print(f"With DSC3k: {output2.shape}")
        With DSC3k: torch.Size([2, 64, 128, 128])
    """
    def __init__(
        self,
        c1,          
        c2,         
        n=1,          
        dsc3k=False,  
        e=0.5,       
        g=1,        
        shortcut=True,
        k1=3,       
        k2=7,       
        d2=1         
    ):
        super().__init__(c1, c2, n, shortcut, g, e)
        if dsc3k:
            self.m = nn.ModuleList(
                DSC3k(
                    self.c, self.c,
                    n=2,           
                    shortcut=shortcut,
                    g=g,
                    e=1.0,  
                    k1=k1,
                    k2=k2,
                    d2=d2
                )
                for _ in range(n)
            )
        else:
            self.m = nn.ModuleList(
                DSBottleneck(
                    self.c, self.c,
                    shortcut=shortcut,
                    e=1.0,
                    k1=k1,
                    k2=k2,
                    d2=d2
                )
                for _ in range(n)
            )


class DRSC(nn.Module):
    """Degradation-residual shallow calibration with a bounded zero-start residual."""

    def __init__(
        self,
        channels: int,
        reduction: int = 4,
        max_gain: float = 0.10,
        pool_kernel: int = 5,
        eps: float = 1e-6,
    ):
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}.")
        if reduction < 1:
            raise ValueError(f"reduction must be at least 1, got {reduction}.")
        if not 0.0 <= max_gain <= 1.0:
            raise ValueError(f"max_gain must be in [0, 1], got {max_gain}.")
        if pool_kernel <= 0 or pool_kernel % 2 == 0:
            raise ValueError(f"pool_kernel must be a positive odd integer, got {pool_kernel}.")
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}.")

        channels = int(channels)
        hidden = max(channels // int(reduction), 16)
        self.channels = channels
        self.pool_kernel = int(pool_kernel)
        self.max_gain = float(max_gain)
        self.eps = float(eps)
        self.alpha_raw = nn.Parameter(torch.zeros(1, dtype=torch.float32))

        self.channel_gate = nn.Sequential(
            nn.Conv2d(2 * channels, hidden, 1, bias=True),
            nn.SiLU(),
            nn.Conv2d(hidden, channels, 1, bias=True),
            nn.Sigmoid(),
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(4, 8, 3, padding=1, bias=True),
            nn.SiLU(),
            nn.Conv2d(8, 1, 1, bias=True),
            nn.Sigmoid(),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 1, bias=True),
        )

    def _lowpass(self, x: torch.Tensor) -> torch.Tensor:
        """Average-pool with replicated boundaries so constant maps remain constant at every pixel."""
        pad = self.pool_kernel // 2
        if pad:
            x = F.pad(x, (pad, pad, pad, pad), mode="replicate")
        return F.avg_pool2d(x, kernel_size=self.pool_kernel, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError(f"DRSC expected [B, {self.channels}, H, W], got {tuple(x.shape)}.")

        low = self._lowpass(x)
        residual = x - low
        abs_residual = residual.abs()

        low_token = F.adaptive_avg_pool2d(low, 1)
        res_token = F.adaptive_avg_pool2d(abs_residual, 1)
        channel_gate = self.channel_gate(torch.cat((low_token, res_token), dim=1))

        low_mean = low.mean(dim=1, keepdim=True)
        low_std = (low.float() - low_mean.float()).pow(2).mean(dim=1, keepdim=True).add(self.eps).sqrt()
        low_std = low_std.to(dtype=x.dtype)
        res_mean = abs_residual.mean(dim=1, keepdim=True)
        res_max = abs_residual.amax(dim=1, keepdim=True)
        spatial_descriptor = torch.cat((low_mean, low_std, res_mean, res_max), dim=1)
        spatial_gate = self.spatial_gate(spatial_descriptor)

        delta = self.refine(residual)
        gain = (self.max_gain * torch.tanh(self.alpha_raw)).to(dtype=x.dtype)
        return x + gain * channel_gate * spatial_gate * delta


class DRSCDSC3k2(DSC3k2):
    """DSC3k2 with an identity-initialized DRSC calibration at its output."""

    def __init__(
        self,
        c1,
        c2,
        n=1,
        dsc3k=False,
        e=0.5,
        drsc_reduction=4,
        drsc_gain=0.10,
        drsc_pool=5,
        g=1,
        shortcut=True,
        k1=3,
        k2=7,
        d2=1,
    ):
        super().__init__(
            c1=c1,
            c2=c2,
            n=n,
            dsc3k=dsc3k,
            e=e,
            g=g,
            shortcut=shortcut,
            k1=k1,
            k2=k2,
            d2=d2,
        )
        self.drsc = DRSC(
            channels=c2,
            reduction=drsc_reduction,
            max_gain=drsc_gain,
            pool_kernel=drsc_pool,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drsc(super().forward(x))


class DAPD(Conv):
    """Degradation-aware polyphase detail-preserving downsampling."""

    def __init__(
        self,
        c1,
        c2,
        k=3,
        s=2,
        p=1,
        g=1,
        max_gain=0.10,
        reduction=4,
        eps=1e-6,
        act=True,
    ):
        if int(c1) <= 0 or int(c2) <= 0:
            raise ValueError(f"DAPD channels must be positive, got {c1}->{c2}.")
        if int(s) != 2:
            raise ValueError(f"DAPD only supports stride=2, got stride={s}.")
        if float(max_gain) < 0.0:
            raise ValueError(f"max_gain must be non-negative, got {max_gain}.")
        if int(reduction) < 1:
            raise ValueError(f"reduction must be >= 1, got {reduction}.")
        if float(eps) <= 0.0:
            raise ValueError(f"eps must be positive, got {eps}.")

        # Keep Conv's parameter names so original layer-3 weights transfer directly.
        super().__init__(c1, c2, k=k, s=s, p=p, g=g, act=act)
        self.c1 = int(c1)
        self.c2 = int(c2)
        self.max_gain = float(max_gain)
        self.eps = float(eps)
        hidden = max(self.c2 // int(reduction), 16)

        self.detail_proj = Conv(4 * self.c1, self.c2, 1, 1, act=False)
        blur = torch.tensor(
            ((1.0, 2.0, 1.0), (2.0, 4.0, 2.0), (1.0, 2.0, 1.0)), dtype=torch.float32
        )
        blur /= blur.sum()
        self.register_buffer("blur_kernel", blur[None, None], persistent=False)
        self.low_proj = Conv(self.c1, self.c2, 1, 1, act=False)

        self.threshold = nn.Sequential(
            nn.Conv2d(self.c2, self.c2, 3, 1, 1, groups=self.c2, bias=False),
            nn.BatchNorm2d(self.c2),
            nn.SiLU(inplace=True),
            nn.Conv2d(self.c2, self.c2, 1, 1, 0, bias=True),
        )
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(2 * self.c2, hidden, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, self.c2, 1, bias=True),
            nn.Sigmoid(),
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(2, 8, 3, 1, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(8, 1, 1, bias=True),
            nn.Sigmoid(),
        )
        self.out_proj = nn.Conv2d(self.c2, self.c2, 1, bias=True)
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.out_proj.bias)

        # The new branch is initially an exact identity relative to Conv.
        self.alpha_raw = nn.Parameter(torch.zeros(1, dtype=torch.float32))

    @staticmethod
    def _pad_even(x):
        """Replicate-pad the right and bottom sides before PixelUnshuffle."""
        pad_h = x.shape[-2] % 2
        pad_w = x.shape[-1] % 2
        return F.pad(x, (0, pad_w, 0, pad_h), mode="replicate") if pad_h or pad_w else x

    def _phase_residual(self, x_even):
        """Return four residual spatial phases ordered as [B, C, 4, H, W]."""
        phase = F.pixel_unshuffle(x_even, downscale_factor=2)
        batch, _, height, width = phase.shape
        phase = phase.view(batch, self.c1, 4, height, width)
        phase = phase - phase.mean(dim=2, keepdim=True)
        return phase.reshape(batch, 4 * self.c1, height, width)

    def _blur_downsample(self, x_even):
        """Create a fixed anti-aliased low-frequency reference."""
        channels = x_even.shape[1]
        kernel = self.blur_kernel.to(device=x_even.device, dtype=x_even.dtype).repeat(channels, 1, 1, 1)
        return F.conv2d(F.pad(x_even, (1, 1, 1, 1), mode="replicate"), kernel, stride=2, groups=channels)

    def _forward_impl(self, x, fused=False):
        if x.ndim != 4:
            raise ValueError(f"DAPD expects NCHW input, got shape={tuple(x.shape)}.")
        base = self.act(self.conv(x)) if fused else self.act(self.bn(self.conv(x)))

        x_even = self._pad_even(x)
        detail = self.detail_proj(self._phase_residual(x_even))
        low = self.low_proj(self._blur_downsample(x_even))
        if detail.shape[-2:] != base.shape[-2:]:
            detail = F.interpolate(detail, size=base.shape[-2:], mode="nearest")
        if low.shape[-2:] != base.shape[-2:]:
            low = F.interpolate(low, size=base.shape[-2:], mode="nearest")

        threshold = 0.10 * F.softplus(self.threshold(low))
        filtered = torch.sign(detail) * F.relu(detail.abs() - threshold)
        channel_gate = self.channel_gate(torch.cat((low, filtered.abs()), dim=1))
        spatial_gate = self.spatial_gate(
            torch.cat(
                (low.abs().mean(dim=1, keepdim=True), filtered.abs().mean(dim=1, keepdim=True)), dim=1
            )
        )
        delta = self.out_proj(filtered * channel_gate * spatial_gate)
        return base + self.max_gain * torch.tanh(self.alpha_raw) * delta

    def forward(self, x):
        return self._forward_impl(x, fused=False)

    def forward_fuse(self, x):
        """Keep the DAPD branch active after BaseModel.fuse() removes BatchNorm."""
        return self._forward_impl(x, fused=True)


class ContourGuidedAdaptiveGeometry(nn.Module):
    """Contour-conditioned, bounded multi-point feature resampling."""

    def __init__(self, channels, samples=5, max_offset=2.0, max_gain=0.08, reduction=4, eps=1e-6):
        super().__init__()
        channels, samples, reduction = int(channels), int(samples), int(reduction)
        max_offset, max_gain, eps = float(max_offset), float(max_gain), float(eps)
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}.")
        if samples < 3 or samples % 2 == 0:
            raise ValueError(f"samples must be an odd integer >= 3, got {samples}.")
        if max_offset < 1.0:
            raise ValueError(f"max_offset must be >= 1.0, got {max_offset}.")
        if max_gain < 0.0 or reduction < 1 or eps <= 0.0:
            raise ValueError("max_gain must be non-negative, reduction >= 1, and eps positive.")

        self.channels, self.samples = channels, samples
        self.max_offset, self.max_gain, self.eps = max_offset, max_gain, eps
        hidden = max(channels // reduction, 16)
        self.reduce = Conv(channels, hidden, 1, 1)
        self.geometry_trunk = nn.Sequential(
            Conv(hidden + 3, hidden, 3, 1), Conv(hidden, hidden, 3, 1, g=hidden), Conv(hidden, hidden, 1, 1)
        )
        self.offset_head = nn.Conv2d(hidden, 2 * samples, 1, bias=True)
        self.weight_head = nn.Conv2d(hidden, samples, 1, bias=True)
        nn.init.zeros_(self.offset_head.weight)
        nn.init.zeros_(self.offset_head.bias)
        nn.init.zeros_(self.weight_head.weight)
        nn.init.zeros_(self.weight_head.bias)
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=True),
        )
        nn.init.normal_(self.refine[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.refine[-1].bias)

        sobel_x = torch.tensor(((-1.0, 0.0, 1.0), (-2.0, 0.0, 2.0), (-1.0, 0.0, 1.0)), dtype=torch.float32)
        self.register_buffer("sobel_x", sobel_x[None, None], persistent=False)
        self.register_buffer("sobel_y", sobel_x.t().contiguous()[None, None], persistent=False)
        base_offsets = [(0.0, 0.0)] + [
            (math.cos(2.0 * math.pi * index / (samples - 1)), math.sin(2.0 * math.pi * index / (samples - 1)))
            for index in range(samples - 1)
        ]
        self.register_buffer("base_offsets", torch.tensor(base_offsets, dtype=torch.float32), persistent=False)
        self.alpha_raw = nn.Parameter(torch.zeros(1, dtype=torch.float32))

    def _contour_cues(self, x):
        summary = F.pad(x.float().mean(dim=1, keepdim=True), (1, 1, 1, 1), mode="replicate")
        grad_x = F.conv2d(summary, self.sobel_x.to(device=x.device, dtype=summary.dtype))
        grad_y = F.conv2d(summary, self.sobel_y.to(device=x.device, dtype=summary.dtype))
        rms = (grad_x.square() + grad_y.square()).mean(dim=(2, 3), keepdim=True).add(self.eps).sqrt()
        grad_x, grad_y = grad_x / rms, grad_y / rms
        magnitude = (grad_x.square() + grad_y.square() + self.eps).sqrt()
        return grad_x.to(x.dtype), grad_y.to(x.dtype), magnitude.to(x.dtype)

    @staticmethod
    def _base_grid(batch, height, width, device):
        y = (torch.arange(height, device=device, dtype=torch.float32) + 0.5) * (2.0 / max(height, 1)) - 1.0
        x = (torch.arange(width, device=device, dtype=torch.float32) + 0.5) * (2.0 / max(width, 1)) - 1.0
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        return torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0).expand(batch, -1, -1, -1)

    def _predict_geometry(self, x):
        grad_x, grad_y, magnitude = self._contour_cues(x)
        geometry = self.geometry_trunk(torch.cat((self.reduce(x), grad_x, grad_y, magnitude), dim=1))
        batch, _, height, width = x.shape
        offset_raw = self.offset_head(geometry).view(batch, self.samples, 2, height, width)
        learned = (self.max_offset - 1.0) * torch.tanh(offset_raw)
        base = self.base_offsets.to(device=x.device, dtype=x.dtype).view(1, self.samples, 2, 1, 1)
        return base + learned, self.weight_head(geometry).softmax(dim=1)

    def _sample(self, x, offsets):
        batch, channels, height, width = x.shape
        base = self._base_grid(batch, height, width, x.device).unsqueeze(1)
        grid = base + torch.stack(
            (offsets[:, :, 0].float() * (2.0 / max(width, 1)), offsets[:, :, 1].float() * (2.0 / max(height, 1))),
            dim=-1,
        )
        source = x.float().unsqueeze(1).expand(-1, self.samples, -1, -1, -1).reshape(
            batch * self.samples, channels, height, width
        )
        sampled = F.grid_sample(
            source, grid.reshape(batch * self.samples, height, width, 2), mode="bilinear", padding_mode="border", align_corners=False
        )
        return sampled.reshape(batch, self.samples, channels, height, width).to(dtype=x.dtype)

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError(f"ContourGuidedAdaptiveGeometry expects NCHW input, got {tuple(x.shape)}.")
        offsets, weights = self._predict_geometry(x)
        sampled = (self._sample(x, offsets) * weights.unsqueeze(2).to(dtype=x.dtype)).sum(dim=1)
        delta = self.refine(sampled - x)
        return x + self.max_gain * torch.tanh(self.alpha_raw) * delta


class CAGDSC3k2(DSC3k2):
    """DSC3k2 plus zero-start contour-guided geometry compensation."""

    def __init__(
        self,
        c1,
        c2,
        n=1,
        dsc3k=False,
        e=0.5,
        samples=5,
        max_offset=2.0,
        max_gain=0.08,
        reduction=4,
        g=1,
        shortcut=True,
        k1=3,
        k2=7,
        d2=1,
    ):
        super().__init__(c1, c2, n, dsc3k, e, g, shortcut, k1, k2, d2)
        self.dsc3k_enabled = bool(dsc3k)
        self.geometry = ContourGuidedAdaptiveGeometry(c2, samples, max_offset, max_gain, reduction)

    def forward(self, x):
        return self.geometry(super().forward(x))




class ShallowEvidenceRouter(nn.Module):
    """Route anti-aliased shallow P2 evidence into a semantic P3 tensor."""

    def __init__(self, c_shallow, c_out, max_gain=0.08, reduction=4, eps=1e-6):
        super().__init__()
        self.c_shallow, self.c_out = int(c_shallow), int(c_out)
        self.max_gain, self.eps = float(max_gain), float(eps)
        if self.c_shallow <= 0 or self.c_out <= 0:
            raise ValueError(f"channels must be positive, got {c_shallow}->{c_out}.")
        if self.max_gain < 0.0 or int(reduction) < 1 or self.eps <= 0.0:
            raise ValueError("max_gain must be non-negative, reduction >= 1, and eps positive.")

        hidden = max(self.c_out // int(reduction), 16)
        blur = torch.tensor(((1.0, 2.0, 1.0), (2.0, 4.0, 2.0), (1.0, 2.0, 1.0)), dtype=torch.float32)
        self.register_buffer("blur_kernel", (blur / blur.sum())[None, None], persistent=False)
        self.low_proj = Conv(self.c_shallow, self.c_out, 1, 1, act=False)
        self.contrast_proj = Conv(self.c_shallow, self.c_out, 1, 1, act=False)
        self.route_gate = nn.Sequential(
            Conv(3 * self.c_out, hidden, 1, 1),
            nn.Conv2d(hidden, self.c_out, 1, bias=True),
            nn.Sigmoid(),
        )
        self.micro_gate = nn.Sequential(
            nn.Conv2d(3, 8, 3, 1, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(8, 1, 1, bias=True),
            nn.Sigmoid(),
        )
        self.out_proj = nn.Conv2d(self.c_out, self.c_out, 1, bias=True)
        nn.init.kaiming_normal_(self.out_proj.weight, mode="fan_out", nonlinearity="linear")
        nn.init.zeros_(self.out_proj.bias)
        # Zero gain preserves the original DSC3k2 path at initialization.
        self.alpha_raw = nn.Parameter(torch.zeros(1, dtype=torch.float32))

    def _blur_downsample(self, x):
        # Use the declared channel count so the depthwise kernel has a static ONNX shape.
        kernel = self.blur_kernel.to(device=x.device, dtype=x.dtype).repeat(self.c_shallow, 1, 1, 1)
        return F.conv2d(F.pad(x, (1, 1, 1, 1), mode="replicate"), kernel, stride=2, groups=self.c_shallow)

    @staticmethod
    def _align(x, size):
        return x if x.shape[-2:] == size else F.interpolate(x, size=size, mode="bilinear", align_corners=False)

    def forward(self, shallow, semantic):
        if shallow.ndim != 4 or semantic.ndim != 4:
            raise ValueError("ShallowEvidenceRouter expects two NCHW tensors.")
        if shallow.shape[0] != semantic.shape[0] or shallow.shape[1] != self.c_shallow:
            raise ValueError("invalid shallow tensor batch size or channel count.")
        if semantic.shape[1] != self.c_out:
            raise ValueError(f"expected {self.c_out} semantic channels, got {semantic.shape[1]}.")

        target = semantic.shape[-2:]
        low_raw = self._align(self._blur_downsample(shallow), target)
        average = F.adaptive_avg_pool2d(shallow, target)
        contrast_raw = (F.adaptive_max_pool2d(shallow, target) - average).clamp_min(0.0)
        low, contrast = self.low_proj(low_raw), self.contrast_proj(contrast_raw)
        route = self.route_gate(torch.cat((low, contrast, semantic), dim=1))
        micro = self.micro_gate(
            torch.cat(
                (
                    low.abs().mean(dim=1, keepdim=True),
                    contrast.abs().mean(dim=1, keepdim=True),
                    semantic.abs().mean(dim=1, keepdim=True),
                ),
                dim=1,
            )
        )
        delta = self.out_proj(route * (low + contrast))
        gain = self.max_gain * torch.tanh(self.alpha_raw)
        return semantic + gain * delta, micro


class CenterPreservedPartialGeometry(nn.Module):
    """Resample only partial channels while enforcing a center sampling floor."""

    def __init__(
        self,
        channels,
        geom_ratio=0.50,
        samples=5,
        min_radius=0.25,
        max_radius=1.50,
        center_floor=0.50,
        max_gain=0.08,
        reduction=4,
        eps=1e-6,
    ):
        super().__init__()
        self.channels, self.samples = int(channels), int(samples)
        self.min_radius, self.max_radius = float(min_radius), float(max_radius)
        self.center_floor, self.max_gain, self.eps = float(center_floor), float(max_gain), float(eps)
        if self.channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}.")
        if not 0.0 < float(geom_ratio) <= 1.0:
            raise ValueError(f"geom_ratio must be in (0, 1], got {geom_ratio}.")
        if self.samples < 3 or self.samples % 2 == 0:
            raise ValueError(f"samples must be an odd integer >= 3, got {samples}.")
        if not 0.0 <= self.min_radius <= self.max_radius:
            raise ValueError("require 0 <= min_radius <= max_radius.")
        if not 0.0 <= self.center_floor <= 1.0:
            raise ValueError("center_floor must be in [0, 1].")
        if self.max_gain < 0.0 or int(reduction) < 1 or self.eps <= 0.0:
            raise ValueError("max_gain must be non-negative, reduction >= 1, and eps positive.")

        self.geom_channels = min(self.channels, max(int(round(self.channels * float(geom_ratio) / 8.0)) * 8, 8))
        hidden = max(self.geom_channels // int(reduction), 16)
        self.reduce = Conv(self.channels, self.geom_channels, 1, 1)
        self.geometry_trunk = nn.Sequential(
            Conv(self.geom_channels + 3, hidden, 3, 1),
            Conv(hidden, hidden, 3, 1, g=hidden),
            Conv(hidden, hidden, 1, 1),
        )
        self.offset_head = nn.Conv2d(hidden, 2 * self.samples, 1, bias=True)
        self.weight_head = nn.Conv2d(hidden, self.samples, 1, bias=True)
        nn.init.zeros_(self.offset_head.weight)
        nn.init.zeros_(self.offset_head.bias)
        nn.init.zeros_(self.weight_head.weight)
        nn.init.zeros_(self.weight_head.bias)
        self.out_proj = nn.Conv2d(self.geom_channels, self.channels, 1, bias=True)
        nn.init.kaiming_normal_(self.out_proj.weight, mode="fan_out", nonlinearity="linear")
        nn.init.zeros_(self.out_proj.bias)

        sobel_x = torch.tensor(((-1.0, 0.0, 1.0), (-2.0, 0.0, 2.0), (-1.0, 0.0, 1.0)), dtype=torch.float32)
        self.register_buffer("sobel_x", sobel_x[None, None], persistent=False)
        self.register_buffer("sobel_y", sobel_x.t().contiguous()[None, None], persistent=False)
        offsets = [(0.0, 0.0)] + [
            (math.cos(2.0 * math.pi * index / (self.samples - 1)), math.sin(2.0 * math.pi * index / (self.samples - 1)))
            for index in range(self.samples - 1)
        ]
        self.register_buffer("base_offsets", torch.tensor(offsets, dtype=torch.float32), persistent=False)
        self.alpha_raw = nn.Parameter(torch.zeros(1, dtype=torch.float32))

    def _contour_cues(self, x):
        summary = F.pad(x.float().mean(dim=1, keepdim=True), (1, 1, 1, 1), mode="replicate")
        grad_x_raw = F.conv2d(summary, self.sobel_x.to(device=x.device, dtype=summary.dtype))
        grad_y_raw = F.conv2d(summary, self.sobel_y.to(device=x.device, dtype=summary.dtype))
        rms = (grad_x_raw.square() + grad_y_raw.square()).mean(dim=(2, 3), keepdim=True).add(self.eps).sqrt()
        grad_x, grad_y = grad_x_raw / rms, grad_y_raw / rms
        magnitude = (grad_x.square() + grad_y.square() + self.eps).sqrt()
        return grad_x.to(x.dtype), grad_y.to(x.dtype), magnitude.to(x.dtype)

    @staticmethod
    def _base_grid(batch, height, width, device):
        y = (torch.arange(height, device=device, dtype=torch.float32) + 0.5) * (2.0 / max(height, 1)) - 1.0
        x = (torch.arange(width, device=device, dtype=torch.float32) + 0.5) * (2.0 / max(width, 1)) - 1.0
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        return torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0).expand(batch, -1, -1, -1)

    def _predict_geometry(self, x, micro):
        grad_x, grad_y, magnitude = self._contour_cues(x)
        geometry = self.geometry_trunk(torch.cat((x, grad_x, grad_y, magnitude), dim=1))
        batch, _, height, width = x.shape
        offset_raw = self.offset_head(geometry).view(batch, self.samples, 2, height, width)
        if micro.shape[-2:] != (height, width):
            micro = F.interpolate(micro, size=(height, width), mode="bilinear", align_corners=False)
        if micro.shape[0] != batch or micro.shape[1] != 1:
            raise ValueError("micro must have shape [B, 1, H, W].")
        radius = self.min_radius + (self.max_radius - self.min_radius) * (1.0 - micro.clamp(0.0, 1.0))
        base = self.base_offsets.to(device=x.device, dtype=offset_raw.dtype).view(1, self.samples, 2, 1, 1)
        residual_limit = 0.5 * (self.max_radius - self.min_radius)
        offsets = (base * radius.unsqueeze(1) + residual_limit * torch.tanh(offset_raw)).clamp(
            -self.max_radius, self.max_radius
        )
        soft_weights = self.weight_head(geometry).float().softmax(dim=1).to(dtype=geometry.dtype)
        center_prior = torch.cat((torch.ones_like(soft_weights[:, :1]), torch.zeros_like(soft_weights[:, 1:])), dim=1)
        weights = (1.0 - self.center_floor) * soft_weights + self.center_floor * center_prior
        return offsets, weights

    def _sample(self, x, offsets):
        batch, channels, height, width = x.shape
        base = self._base_grid(batch, height, width, x.device).unsqueeze(1)
        grid = base + torch.stack(
            (
                offsets[:, :, 0].float() * (2.0 / max(width, 1)),
                offsets[:, :, 1].float() * (2.0 / max(height, 1)),
            ),
            dim=-1,
        )
        source = x.float().unsqueeze(1).expand(-1, self.samples, -1, -1, -1).reshape(
            batch * self.samples, channels, height, width
        )
        sampled = F.grid_sample(
            source,
            grid.reshape(batch * self.samples, height, width, 2),
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        return sampled.reshape(batch, self.samples, channels, height, width).to(dtype=x.dtype)

    def forward(self, x, micro):
        if x.ndim != 4 or micro.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError(f"expected x=[B,{self.channels},H,W] and micro=[B,1,H,W].")
        geometry_feature = self.reduce(x)
        offsets, weights = self._predict_geometry(geometry_feature, micro)
        sampled = (self._sample(geometry_feature, offsets) * weights.unsqueeze(2).to(geometry_feature.dtype)).sum(dim=1)
        delta = self.out_proj(sampled - geometry_feature)
        return x + self.max_gain * torch.tanh(self.alpha_raw) * delta


class SCPGDSC3k2(DSC3k2):
    """Original DSC3k2 main path plus SER and center-preserved partial geometry."""

    def __init__(
        self,
        c_shallow,
        c_p3,
        c2,
        n=1,
        dsc3k=False,
        e=0.25,
        geom_ratio=0.50,
        samples=5,
        min_radius=0.25,
        max_radius=1.50,
        center_floor=0.50,
        detail_gain=0.08,
        geom_gain=0.08,
        reduction=4,
        g=1,
        shortcut=True,
        k1=3,
        k2=7,
        d2=1,
    ):
        super().__init__(c1=c_p3, c2=c2, n=n, dsc3k=dsc3k, e=e, g=g, shortcut=shortcut, k1=k1, k2=k2, d2=d2)
        self.c_shallow, self.c_p3, self.c2 = int(c_shallow), int(c_p3), int(c2)
        self.dsc3k_enabled = bool(dsc3k)
        # Keep downstream same-seed initialization identical to the original DSC3k2 path.
        with torch.random.fork_rng(devices=[], enabled=True):
            self.detail_router = ShallowEvidenceRouter(
                self.c_shallow, self.c2, max_gain=detail_gain, reduction=reduction
            )
            self.geometry = CenterPreservedPartialGeometry(
                self.c2,
                geom_ratio=geom_ratio,
                samples=samples,
                min_radius=min_radius,
                max_radius=max_radius,
                center_floor=center_floor,
                max_gain=geom_gain,
                reduction=reduction,
            )

    def forward(self, x):
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise ValueError("SCPGDSC3k2 expects [P2_shallow, P3_downsampled].")
        shallow, p3 = x
        if shallow.ndim != 4 or p3.ndim != 4 or shallow.shape[0] != p3.shape[0]:
            raise ValueError("SCPGDSC3k2 inputs must be compatible NCHW tensors.")
        if shallow.shape[1] != self.c_shallow or p3.shape[1] != self.c_p3:
            raise ValueError(
                f"expected P2/P3 channels {self.c_shallow}/{self.c_p3}, got {shallow.shape[1]}/{p3.shape[1]}."
            )
        base = super().forward(p3)
        routed, micro = self.detail_router(shallow, base)
        return self.geometry(routed, micro)


class ConsensusBudgetedEvidenceRouter(ShallowEvidenceRouter):
    """Release shallow evidence only in consensus-supported regions under a channel budget."""

    def __init__(
        self,
        c_shallow,
        c_out,
        max_gain=0.08,
        reduction=4,
        release_rho=0.20,
        local_self=0.75,
        co_weight=0.50,
        eps=1e-6,
    ):
        super().__init__(c_shallow=c_shallow, c_out=c_out, max_gain=max_gain, reduction=reduction, eps=eps)
        self.release_rho, self.local_self, self.co_weight = float(release_rho), float(local_self), float(co_weight)
        if not 0.0 <= self.release_rho <= 1.0:
            raise ValueError(f"release_rho must be in [0, 1], got {release_rho}.")
        if not 0.0 <= self.local_self <= 1.0:
            raise ValueError(f"local_self must be in [0, 1], got {local_self}.")
        if not 0.0 <= self.co_weight <= 1.0:
            raise ValueError(f"co_weight must be in [0, 1], got {co_weight}.")

        hidden = max(self.c_out // int(reduction), 16)
        self.spatial_compatibility = nn.Sequential(
            nn.Conv2d(4, 8, 3, 1, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(8, 1, 1, 1, 0, bias=True),
        )
        self.channel_compatibility = nn.Sequential(
            nn.Conv2d(3 * self.c_out, hidden, 1, 1, 0, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, self.c_out, 1, 1, 0, bias=True),
        )
        # alpha_raw remains zero, preserving the exact H2/H3 initialization. Small nonzero heads avoid dead gates.
        nn.init.normal_(self.spatial_compatibility[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.spatial_compatibility[-1].bias)
        nn.init.normal_(self.channel_compatibility[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.channel_compatibility[-1].bias)
        self.latest_diagnostics = {}

    @staticmethod
    def _local_mean(x):
        """Return a 3x3 local mean without zero-padding boundary bias."""
        return F.avg_pool2d(F.pad(x, (1, 1, 1, 1), mode="replicate"), kernel_size=3, stride=1, padding=0)

    @staticmethod
    def _spatial_zscore(x, eps):
        value = x.float()
        mean = value.mean(dim=(2, 3), keepdim=True)
        std = value.var(dim=(2, 3), keepdim=True, unbiased=False).add(eps).sqrt()
        return (value - mean) / std

    def _relative_support(self, detail_energy, low_energy):
        """Combine relative saliency with absolute SNR so a flat map has zero support, not 0.5."""
        ratio = (detail_energy.float() / (low_energy.float() + self.eps)).clamp(0.0, 8.0)
        absolute_support = 1.0 - torch.exp(-ratio)
        relative_support = torch.sigmoid(self._spatial_zscore(detail_energy, self.eps))
        return (absolute_support * relative_support).clamp(0.0, 1.0)

    @staticmethod
    def _normalize_channel_token(token, eps):
        return token / (token.mean(dim=1, keepdim=True) + eps)

    def _budgeted_channel_gate(self, low, contrast, semantic_detail):
        """Return a [B,C,1,1] gate whose per-sample mean cannot exceed release_rho."""
        low_token = low.float().abs().mean(dim=(2, 3), keepdim=True)
        contrast_token = contrast.float().abs().mean(dim=(2, 3), keepdim=True)
        semantic_token = semantic_detail.float().abs().mean(dim=(2, 3), keepdim=True)
        detail_share = contrast_token / (contrast_token + semantic_token + self.eps)
        shallow_unique = (2.0 * detail_share - 1.0).clamp_min(0.0)
        coactivated = (1.0 - (2.0 * detail_share - 1.0).abs()).clamp(0.0, 1.0)
        channel_prior = (shallow_unique + self.co_weight * coactivated).clamp(0.0, 1.0)
        channel_tokens = torch.cat(
            (
                self._normalize_channel_token(low_token, self.eps),
                self._normalize_channel_token(contrast_token, self.eps),
                self._normalize_channel_token(semantic_token, self.eps),
            ),
            dim=1,
        ).detach().to(dtype=low.dtype)
        learned_gate = torch.sigmoid(self.channel_compatibility(channel_tokens)).float()
        raw_gate = learned_gate * channel_prior.detach()
        raw_mean = raw_gate.mean(dim=1, keepdim=True)
        budget_scale = torch.where(
            raw_mean > self.eps,
            torch.full_like(raw_mean, self.release_rho) / (raw_mean.detach() + self.eps),
            torch.zeros_like(raw_mean),
        )
        return (raw_gate * budget_scale).clamp(0.0, 1.0).to(dtype=low.dtype)

    def forward(self, shallow, semantic):
        if shallow.ndim != 4 or semantic.ndim != 4:
            raise ValueError("ConsensusBudgetedEvidenceRouter expects two NCHW tensors.")
        if shallow.shape[0] != semantic.shape[0]:
            raise ValueError("shallow and semantic must have the same batch size.")
        if shallow.shape[1] != self.c_shallow:
            raise ValueError(f"expected {self.c_shallow} shallow channels, got {shallow.shape[1]}.")
        if semantic.shape[1] != self.c_out:
            raise ValueError(f"expected {self.c_out} semantic channels, got {semantic.shape[1]}.")

        target_size = semantic.shape[-2:]
        low_raw = self._align(self._blur_downsample(shallow), target_size)
        average_raw = F.adaptive_avg_pool2d(shallow, target_size)
        contrast_raw = (F.adaptive_max_pool2d(shallow, target_size) - average_raw).clamp_min(0.0)
        low, contrast = self.low_proj(low_raw), self.contrast_proj(contrast_raw)
        route = self.route_gate(torch.cat((low, contrast, semantic), dim=1))
        micro = self.micro_gate(
            torch.cat(
                (
                    low.abs().mean(dim=1, keepdim=True),
                    contrast.abs().mean(dim=1, keepdim=True),
                    semantic.abs().mean(dim=1, keepdim=True),
                ),
                dim=1,
            )
        )

        shallow_detail_energy = contrast_raw.float().abs().mean(dim=1, keepdim=True)
        shallow_low_energy = low_raw.float().abs().mean(dim=1, keepdim=True)
        shallow_support = self._relative_support(shallow_detail_energy, shallow_low_energy)
        semantic_float = semantic.float()
        semantic_low = self._local_mean(semantic_float)
        semantic_detail = (semantic_float - semantic_low).abs()
        semantic_detail_energy = semantic_detail.mean(dim=1, keepdim=True)
        semantic_low_energy = semantic_low.abs().mean(dim=1, keepdim=True)
        semantic_support = self._relative_support(semantic_detail_energy, semantic_low_energy)
        coactivated = shallow_support * semantic_support
        shallow_unique = shallow_support * (1.0 - semantic_support)
        raw_prior = (shallow_unique + self.co_weight * coactivated).clamp(0.0, 1.0)
        local_prior = (
            self.local_self * raw_prior + (1.0 - self.local_self) * self._local_mean(raw_prior)
        ).clamp(0.0, 1.0)
        spatial_cues = torch.cat((shallow_support, semantic_support, shallow_unique, coactivated), dim=1).detach().to(
            dtype=semantic.dtype
        )
        learned_spatial = torch.sigmoid(self.spatial_compatibility(spatial_cues))
        spatial_gate = local_prior.detach().to(dtype=semantic.dtype) * learned_spatial
        channel_gate = self._budgeted_channel_gate(low, contrast, semantic_detail.to(dtype=semantic.dtype))
        delta = self.out_proj(route * (low + contrast))
        gain = (self.max_gain * torch.tanh(self.alpha_raw)).to(dtype=semantic.dtype)
        routed = semantic + gain * spatial_gate * channel_gate * delta
        self.latest_diagnostics = {
            "spatial_gate": spatial_gate.detach(),
            "channel_gate": channel_gate.detach(),
            "gain": gain.detach(),
            "local_prior": local_prior.detach(),
        }
        return routed, micro


class CBERSCPGDSC3k2(SCPGDSC3k2):
    """SCPG-H3 whose SER residual is routed by local consensus and a channel budget."""

    def __init__(
        self,
        c_shallow,
        c_p3,
        c2,
        n=1,
        dsc3k=False,
        e=0.25,
        geom_ratio=0.50,
        samples=5,
        min_radius=0.25,
        max_radius=1.50,
        center_floor=0.50,
        detail_gain=0.08,
        geom_gain=0.08,
        reduction=4,
        release_rho=0.20,
        local_self=0.75,
        co_weight=0.50,
        g=1,
        shortcut=True,
        k1=3,
        k2=7,
        d2=1,
    ):
        super().__init__(
            c_shallow=c_shallow,
            c_p3=c_p3,
            c2=c2,
            n=n,
            dsc3k=dsc3k,
            e=e,
            geom_ratio=geom_ratio,
            samples=samples,
            min_radius=min_radius,
            max_radius=max_radius,
            center_floor=center_floor,
            detail_gain=detail_gain,
            geom_gain=geom_gain,
            reduction=reduction,
            g=g,
            shortcut=shortcut,
            k1=k1,
            k2=k2,
            d2=d2,
        )
        self.detail_router = ConsensusBudgetedEvidenceRouter(
            c_shallow=self.c_shallow,
            c_out=self.c2,
            max_gain=detail_gain,
            reduction=reduction,
            release_rho=release_rho,
            local_self=local_self,
            co_weight=co_weight,
        )


class AdaHyperedgeGen(nn.Module):
    """
    Generates an adaptive hyperedge participation matrix from a set of vertex features.

    This module implements the Adaptive Hyperedge Generation mechanism. It generates dynamic hyperedge prototypes
    based on the global context of the input nodes and calculates a continuous participation matrix (A)
    that defines the relationship between each vertex and each hyperedge.

    Attributes:
        node_dim (int): The feature dimension of each input node.
        num_hyperedges (int): The number of hyperedges to generate.
        num_heads (int, optional): The number of attention heads for multi-head similarity calculation. Defaults to 4.
        dropout (float, optional): The dropout rate applied to the logits. Defaults to 0.1.
        context (str, optional): The type of global context to use ('mean', 'max', or 'both'). Defaults to "both".

    Methods:
        forward: Takes a batch of vertex features and returns the participation matrix A.

    Examples:
        >>> import torch
        >>> model = AdaHyperedgeGen(node_dim=64, num_hyperedges=16, num_heads=4)
        >>> x = torch.randn(2, 100, 64)  # (Batch, Num_Nodes, Node_Dim)
        >>> A = model(x)
        >>> print(A.shape)
        torch.Size([2, 100, 16])
    """
    def __init__(self, node_dim, num_hyperedges, num_heads=4, dropout=0.1, context="both"):
        super().__init__()
        self.num_heads = num_heads
        self.num_hyperedges = num_hyperedges
        self.head_dim = node_dim // num_heads
        self.context = context

        self.prototype_base = nn.Parameter(torch.Tensor(num_hyperedges, node_dim))
        nn.init.xavier_uniform_(self.prototype_base)
        if context in ("mean", "max"):
            self.context_net = nn.Linear(node_dim, num_hyperedges * node_dim)  
        elif context == "both":
            self.context_net = nn.Linear(2*node_dim, num_hyperedges * node_dim)
        else:
            raise ValueError(
                f"Unsupported context '{context}'. "
                "Expected one of: 'mean', 'max', 'both'."
            )

        self.pre_head_proj = nn.Linear(node_dim, node_dim)
    
        self.dropout = nn.Dropout(dropout)
        self.scaling = math.sqrt(self.head_dim)

    def forward(self, X):
        B, N, D = X.shape
        if self.context == "mean":
            context_cat = X.mean(dim=1)          
        elif self.context == "max":
            context_cat, _ = X.max(dim=1)          
        else:
            avg_context = X.mean(dim=1)           
            max_context, _ = X.max(dim=1)           
            context_cat = torch.cat([avg_context, max_context], dim=-1) 
        prototype_offsets = self.context_net(context_cat).view(B, self.num_hyperedges, D)  
        prototypes = self.prototype_base.unsqueeze(0) + prototype_offsets           
        
        X_proj = self.pre_head_proj(X) 
        X_heads = X_proj.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        proto_heads = prototypes.view(B, self.num_hyperedges, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        X_heads_flat = X_heads.reshape(B * self.num_heads, N, self.head_dim)
        proto_heads_flat = proto_heads.reshape(B * self.num_heads, self.num_hyperedges, self.head_dim).transpose(1, 2)
        
        logits = torch.bmm(X_heads_flat, proto_heads_flat) / self.scaling 
        logits = logits.view(B, self.num_heads, N, self.num_hyperedges).mean(dim=1) 
        
        logits = self.dropout(logits)  

        return F.softmax(logits, dim=1)

class AdaHGConv(nn.Module):
    """
    Performs the adaptive hypergraph convolution.

    This module contains the two-stage message passing process of hypergraph convolution:
    1. Generates an adaptive participation matrix using AdaHyperedgeGen.
    2. Aggregates vertex features into hyperedge features (vertex-to-edge).
    3. Disseminates hyperedge features back to update vertex features (edge-to-vertex).
    A residual connection is added to the final output.

    Attributes:
        embed_dim (int): The feature dimension of the vertices.
        num_hyperedges (int, optional): The number of hyperedges for the internal generator. Defaults to 16.
        num_heads (int, optional): The number of attention heads for the internal generator. Defaults to 4.
        dropout (float, optional): The dropout rate for the internal generator. Defaults to 0.1.
        context (str, optional): The context type for the internal generator. Defaults to "both".

    Methods:
        forward: Performs the adaptive hypergraph convolution on a batch of vertex features.

    Examples:
        >>> import torch
        >>> model = AdaHGConv(embed_dim=128, num_hyperedges=16, num_heads=8)
        >>> x = torch.randn(2, 256, 128) # (Batch, Num_Nodes, Dim)
        >>> output = model(x)
        >>> print(output.shape)
        torch.Size([2, 256, 128])
    """
    def __init__(self, embed_dim, num_hyperedges=16, num_heads=4, dropout=0.1, context="both"):
        super().__init__()
        self.edge_generator = AdaHyperedgeGen(embed_dim, num_hyperedges, num_heads, dropout, context)
        self.edge_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim ),
            nn.GELU()
        )
        self.node_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim ),
            nn.GELU()
        )
        
    def forward(self, X):
        A = self.edge_generator(X)  
        
        He = torch.bmm(A.transpose(1, 2), X) 
        He = self.edge_proj(He)
        
        X_new = torch.bmm(A, He)  
        X_new = self.node_proj(X_new)
        
        return X_new + X
        
class AdaHGComputation(nn.Module):
    """
    A wrapper module for applying adaptive hypergraph convolution to 4D feature maps.

    This class makes the hypergraph convolution compatible with standard CNN architectures. It flattens a
    4D input tensor (B, C, H, W) into a sequence of vertices (tokens), applies the AdaHGConv layer to
    model high-order correlations, and then reshapes the output back into a 4D tensor.

    Attributes:
        embed_dim (int): The feature dimension of the vertices (equivalent to input channels C).
        num_hyperedges (int, optional): The number of hyperedges for the underlying AdaHGConv. Defaults to 16.
        num_heads (int, optional): The number of attention heads for the underlying AdaHGConv. Defaults to 8.
        dropout (float, optional): The dropout rate for the underlying AdaHGConv. Defaults to 0.1.
        context (str, optional): The context type for the underlying AdaHGConv. Defaults to "both".

    Methods:
        forward: Processes a 4D feature map through the adaptive hypergraph computation layer.

    Examples:
        >>> import torch
        >>> model = AdaHGComputation(embed_dim=64, num_hyperedges=8, num_heads=4)
        >>> x = torch.randn(2, 64, 32, 32) # (B, C, H, W)
        >>> output = model(x)
        >>> print(output.shape)
        torch.Size([2, 64, 32, 32])
    """
    def __init__(self, embed_dim, num_hyperedges=16, num_heads=8, dropout=0.1, context="both"):
        super().__init__()
        self.embed_dim = embed_dim
        self.hgnn = AdaHGConv(
            embed_dim=embed_dim,
            num_hyperedges=num_hyperedges,
            num_heads=num_heads,
            dropout=dropout,
            context=context
        )
        
    def forward(self, x):
        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2) 
        tokens = self.hgnn(tokens) 
        x_out = tokens.transpose(1, 2).view(B, C, H, W)
        return x_out 

class C3AH(nn.Module):
    """
    A CSP-style block integrating Adaptive Hypergraph Computation (C3AH).

    The input feature map is split into two paths.
    One path is processed by the AdaHGComputation module to model high-order correlations, while the other
    serves as a shortcut. The outputs are then concatenated to fuse features.

    Attributes:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        e (float, optional): Expansion ratio for the hidden channels. Defaults to 1.0.
        num_hyperedges (int, optional): The number of hyperedges for the internal AdaHGComputation. Defaults to 8.
        context (str, optional): The context type for the internal AdaHGComputation. Defaults to "both".

    Methods:
        forward: Performs a forward pass through the C3AH module.

    Examples:
        >>> import torch
        >>> model = C3AH(c1=64, c2=128, num_hyperedges=8)
        >>> x = torch.randn(2, 64, 32, 32)
        >>> output = model(x)
        >>> print(output.shape)
        torch.Size([2, 128, 32, 32])
    """
    def __init__(self, c1, c2, e=1.0, num_hyperedges=8, context="both"):
        super().__init__()
        c_ = int(c2 * e)  
        assert c_ % 16 == 0, "Dimension of AdaHGComputation should be a multiple of 16."
        num_heads = c_ // 16
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.m = AdaHGComputation(embed_dim=c_, 
                          num_hyperedges=num_hyperedges, 
                          num_heads=num_heads,
                          dropout=0.1,
                          context=context)
        self.cv3 = Conv(2 * c_, c2, 1)  
        
    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))

class FuseModule(nn.Module):
    """
    A module to fuse multi-scale features for the HyperACE block.

    This module takes a list of three feature maps from different scales, aligns them to a common
    spatial resolution by downsampling the first and upsampling the third, and then concatenates
    and fuses them with a convolution layer.

    Attributes:
        c_in (int): The number of channels of the input feature maps.
        channel_adjust (bool): Whether to adjust the channel count of the concatenated features.

    Methods:
        forward: Fuses a list of three multi-scale feature maps.

    Examples:
        >>> import torch
        >>> model = FuseModule(c_in=64, channel_adjust=False)
        >>> # Input is a list of features from different backbone stages
        >>> x_list = [torch.randn(2, 64, 64, 64), torch.randn(2, 64, 32, 32), torch.randn(2, 64, 16, 16)]
        >>> output = model(x_list)
        >>> print(output.shape)
        torch.Size([2, 64, 32, 32])
    """
    def __init__(self, c_in, channel_adjust):
        super(FuseModule, self).__init__()
        self.downsample = nn.AvgPool2d(kernel_size=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        if channel_adjust:
            self.conv_out = Conv(4 * c_in, c_in, 1)
        else:
            self.conv_out = Conv(3 * c_in, c_in, 1)

    def forward(self, x):
        x1_ds = self.downsample(x[0])
        x3_up = self.upsample(x[2])
        x_cat = torch.cat([x1_ds, x[1], x3_up], dim=1)
        out = self.conv_out(x_cat)
        return out

class HyperACE(nn.Module):
    """
    Hypergraph-based Adaptive Correlation Enhancement (HyperACE).

    This is the core module of YOLOv13, designed to model both global high-order correlations and
    local low-order correlations. It first fuses multi-scale features, then processes them through parallel
    branches: two C3AH branches for high-order modeling and a lightweight DSConv-based branch for
    low-order feature extraction.

    Attributes:
        c1 (int): Number of input channels for the fuse module.
        c2 (int): Number of output channels for the entire block.
        n (int, optional): Number of blocks in the low-order branch. Defaults to 1.
        num_hyperedges (int, optional): Number of hyperedges for the C3AH branches. Defaults to 8.
        dsc3k (bool, optional): If True, use DSC3k in the low-order branch; otherwise, use DSBottleneck. Defaults to True.
        shortcut (bool, optional): Whether to use shortcuts in the low-order branch. Defaults to False.
        e1 (float, optional): Expansion ratio for the main hidden channels. Defaults to 0.5.
        e2 (float, optional): Expansion ratio within the C3AH branches. Defaults to 1.
        context (str, optional): Context type for C3AH branches. Defaults to "both".
        channel_adjust (bool, optional): Passed to FuseModule for channel configuration. Defaults to True.

    Methods:
        forward: Performs a forward pass through the HyperACE module.

    Examples:
        >>> import torch
        >>> model = HyperACE(c1=64, c2=256, n=1, num_hyperedges=8)
        >>> x_list = [torch.randn(2, 64, 64, 64), torch.randn(2, 64, 32, 32), torch.randn(2, 64, 16, 16)]
        >>> output = model(x_list)
        >>> print(output.shape)
        torch.Size([2, 256, 32, 32])
    """
    def __init__(self, c1, c2, n=1, num_hyperedges=8, dsc3k=True, shortcut=False, e1=0.5, e2=1, context="both", channel_adjust=True):
        super().__init__()
        self.c = int(c2 * e1) 
        self.cv1 = Conv(c1, 3 * self.c, 1, 1)
        self.cv2 = Conv((4 + n) * self.c, c2, 1) 
        self.m = nn.ModuleList(
            DSC3k(self.c, self.c, 2, shortcut, k1=3, k2=7) if dsc3k else DSBottleneck(self.c, self.c, shortcut=shortcut) for _ in range(n)
        )
        self.fuse = FuseModule(c1, channel_adjust)
        self.branch1 = C3AH(self.c, self.c, e2, num_hyperedges, context)
        self.branch2 = C3AH(self.c, self.c, e2, num_hyperedges, context)
                    
    def forward(self, X):
        x = self.fuse(X)
        y = list(self.cv1(x).chunk(3, 1))
        out1 = self.branch1(y[1])
        out2 = self.branch2(y[1])
        y.extend(m(y[-1]) for m in self.m)
        y[1] = out1
        y.append(out2)
        return self.cv2(torch.cat(y, 1))

class DownsampleConv(nn.Module):
    """
    A simple downsampling block with optional channel adjustment.

    This module uses average pooling to reduce the spatial dimensions (H, W) by a factor of 2. It can
    optionally include a 1x1 convolution to adjust the number of channels, typically doubling them.

    Attributes:
        in_channels (int): The number of input channels.
        channel_adjust (bool, optional): If True, a 1x1 convolution doubles the channel dimension. Defaults to True.

    Methods:
        forward: Performs the downsampling and optional channel adjustment.

    Examples:
        >>> import torch
        >>> model = DownsampleConv(in_channels=64, channel_adjust=True)
        >>> x = torch.randn(2, 64, 32, 32)
        >>> output = model(x)
        >>> print(output.shape)
        torch.Size([2, 128, 16, 16])
    """
    def __init__(self, in_channels, channel_adjust=True):
        super().__init__()
        self.downsample = nn.AvgPool2d(kernel_size=2)
        if channel_adjust:
            self.channel_adjust = Conv(in_channels, in_channels * 2, 1)
        else:
            self.channel_adjust = nn.Identity() 

    def forward(self, x):
        return self.channel_adjust(self.downsample(x))

class FullPAD_Tunnel(nn.Module):
    """
    A gated fusion module for the Full-Pipeline Aggregation-and-Distribution (FullPAD) paradigm.

    This module implements a gated residual connection used to fuse features. It takes two inputs: the original
    feature map and a correlation-enhanced feature map. It then computes `output = original + gate * enhanced`,
    where `gate` is a learnable scalar parameter that adaptively balances the contribution of the enhanced features.

    Methods:
        forward: Performs the gated fusion of two input feature maps.

    Examples:
        >>> import torch
        >>> model = FullPAD_Tunnel()
        >>> original_feature = torch.randn(2, 64, 32, 32)
        >>> enhanced_feature = torch.randn(2, 64, 32, 32)
        >>> output = model([original_feature, enhanced_feature])
        >>> print(output.shape)
        torch.Size([2, 64, 32, 32])
    """
    def __init__(self):
        super().__init__()
        self.gate = nn.Parameter(torch.tensor(0.0))
    def forward(self, x):
        out = x[0] + self.gate * x[1]
        return out


class BCRAUp(nn.Module):
    """Boundary-conditioned, bounded discrete P5-to-P4 reassembly with an exact nearest main path."""

    def __init__(
        self,
        c_deep,
        c_lateral,
        scale=2,
        kernel_size=3,
        align_ratio=0.50,
        reduction=4,
        temperature=0.20,
        residual_groups=4,
        max_residual_ratio=0.20,
        confidence_floor=0.25,
        use_boundary_query=True,
        use_entropy=True,
        use_energy_budget=True,
        detach_confidence=True,
        detach_budget=True,
        strict_scale=True,
        eps=1e-6,
    ):
        super().__init__()
        self.c_deep, self.c_lateral = int(c_deep), int(c_lateral)
        self.scale, self.kernel_size = int(scale), int(kernel_size)
        self.temperature, self.max_residual_ratio = float(temperature), float(max_residual_ratio)
        self.confidence_floor, self.eps = float(confidence_floor), float(eps)
        self.use_boundary_query, self.use_entropy = bool(use_boundary_query), bool(use_entropy)
        self.use_energy_budget = bool(use_energy_budget)
        self.detach_confidence, self.detach_budget, self.strict_scale = (
            bool(detach_confidence),
            bool(detach_budget),
            bool(strict_scale),
        )
        if self.c_deep <= 0 or self.c_lateral <= 0:
            raise ValueError("BCRAUp channel counts must be positive.")
        if self.scale <= 1 or self.kernel_size < 3 or self.kernel_size % 2 == 0:
            raise ValueError("BCRAUp requires scale > 1 and an odd kernel_size >= 3.")
        if not 0.0 < float(align_ratio) <= 1.0 or int(reduction) < 1:
            raise ValueError("align_ratio must be in (0, 1] and reduction must be >= 1.")
        if self.temperature <= 0.0 or not 0.0 < self.max_residual_ratio <= 1.0:
            raise ValueError("temperature and max_residual_ratio must be positive.")
        if not 0.0 <= self.confidence_floor <= 1.0 or self.eps <= 0.0:
            raise ValueError("confidence_floor must be in [0, 1] and eps must be positive.")

        raw_channels = int(round(self.c_deep * float(align_ratio) / 8.0)) * 8
        self.align_channels = min(self.c_deep, max(16, raw_channels))
        self.embed_dim = max(16, min(64, min(self.align_channels, self.c_lateral) // int(reduction)))
        radius = self.kernel_size // 2
        self.offsets = [(dy, dx) for dy in range(-radius, radius + 1) for dx in range(-radius, radius + 1)]
        self.num_candidates = len(self.offsets)

        with torch.random.fork_rng(devices=[], enabled=True):
            self.semantic_proj = nn.Conv2d(self.c_lateral, self.embed_dim, 1, bias=False)
            if self.use_boundary_query:
                sobel_x = torch.tensor(((-1.0, 0.0, 1.0), (-2.0, 0.0, 2.0), (-1.0, 0.0, 1.0)), dtype=torch.float32)
                self.register_buffer("sobel_x", sobel_x[None, None], persistent=False)
                self.register_buffer("sobel_y", sobel_x.t().contiguous()[None, None], persistent=False)
                self.edge_proj = nn.Sequential(
                    nn.Conv2d(3, 8, 3, 1, 1, bias=True), nn.SiLU(inplace=True), nn.Conv2d(8, self.embed_dim, 1, bias=False)
                )
                self.edge_gate = nn.Sequential(
                    nn.Conv2d(2, 8, 3, 1, 1, bias=True), nn.SiLU(inplace=True), nn.Conv2d(8, 1, 1, bias=True), nn.Sigmoid()
                )
                self.query_mix = nn.Conv2d(2 * self.embed_dim, self.embed_dim, 1, bias=False)
            else:
                self.register_buffer("sobel_x", torch.empty(0), persistent=False)
                self.register_buffer("sobel_y", torch.empty(0), persistent=False)
                self.edge_proj = self.edge_gate = self.query_mix = None
            self.key_proj = nn.Conv2d(self.c_deep, self.embed_dim, 1, bias=False)
            self.value_proj = nn.Identity() if self.align_channels == self.c_deep else nn.Conv2d(self.c_deep, self.align_channels, 1, bias=False)
            groups = max(1, math.gcd(math.gcd(self.align_channels, self.c_deep), int(residual_groups)))
            self.residual_out = nn.Conv2d(self.align_channels, self.c_deep, 1, groups=groups, bias=False)
        nn.init.zeros_(self.residual_out.weight)

    def _validate_inputs(self, deep, lateral):
        if deep.ndim != 4 or lateral.ndim != 4 or deep.shape[0] != lateral.shape[0]:
            raise ValueError("BCRAUp expects compatible NCHW deep and lateral tensors.")
        if deep.shape[1] != self.c_deep or lateral.shape[1] != self.c_lateral:
            raise ValueError(f"BCRAUp expected channels {self.c_deep}/{self.c_lateral}, got {deep.shape[1]}/{lateral.shape[1]}.")
        if deep.device != lateral.device or deep.dtype != lateral.dtype:
            raise ValueError("BCRAUp inputs must share device and dtype.")
        expected = (deep.shape[-2] * self.scale, deep.shape[-1] * self.scale)
        if self.strict_scale and tuple(lateral.shape[-2:]) != expected:
            raise ValueError(f"BCRAUp expected lateral size {expected}, got {tuple(lateral.shape[-2:])}.")

    def _edge_cues(self, lateral):
        summary = F.pad(lateral.float().mean(dim=1, keepdim=True), (1, 1, 1, 1), mode="replicate")
        grad_x = F.conv2d(summary, self.sobel_x.to(device=lateral.device, dtype=summary.dtype))
        grad_y = F.conv2d(summary, self.sobel_y.to(device=lateral.device, dtype=summary.dtype))
        rms = (grad_x.square() + grad_y.square()).mean(dim=(2, 3), keepdim=True).add(self.eps).sqrt()
        grad_x, grad_y = grad_x / rms, grad_y / rms
        magnitude = (grad_x.square() + grad_y.square() + self.eps).sqrt()
        return grad_x.to(lateral.dtype), grad_y.to(lateral.dtype), magnitude.to(lateral.dtype)

    def _build_query(self, lateral):
        semantic = self.semantic_proj(lateral)
        if not self.use_boundary_query:
            return semantic
        grad_x, grad_y, magnitude = self._edge_cues(lateral)
        edge = self.edge_proj(torch.cat((grad_x, grad_y, magnitude), dim=1))
        semantic_energy = semantic.float().abs().mean(dim=1, keepdim=True).to(magnitude.dtype)
        gate = self.edge_gate(torch.cat((magnitude, semantic_energy), dim=1))
        return self.query_mix(torch.cat((semantic, gate * edge), dim=1))

    def _shift_replicate(self, feature, delta_y, delta_x):
        radius = self.kernel_size // 2
        padded = F.pad(feature, (radius, radius, radius, radius), mode="replicate")
        height, width = feature.shape[-2:]
        y_start, x_start = radius + int(delta_y), radius + int(delta_x)
        return padded[..., y_start : y_start + height, x_start : x_start + width]

    def _reassemble(self, deep, query):
        target_size = query.shape[-2:]
        query_fp32 = F.normalize(query.float(), dim=1, eps=self.eps)
        key = self.key_proj(deep)
        logits = []
        for delta_y, delta_x in self.offsets:
            shifted = F.interpolate(self._shift_replicate(key, delta_y, delta_x), size=target_size, mode="nearest")
            logits.append((query_fp32 * F.normalize(shifted.float(), dim=1, eps=self.eps)).sum(dim=1))
        weights = (torch.stack(logits, dim=1) / self.temperature).softmax(dim=1)
        entropy = -(weights.clamp_min(self.eps).log() * weights).sum(dim=1, keepdim=True) / math.log(self.num_candidates)
        confidence = (1.0 - entropy).clamp(0.0, 1.0)
        value = self.value_proj(deep)
        reassembled = torch.zeros((deep.shape[0], self.align_channels, *target_size), device=deep.device, dtype=torch.float32)
        for index, (delta_y, delta_x) in enumerate(self.offsets):
            shifted = F.interpolate(self._shift_replicate(value, delta_y, delta_x), size=target_size, mode="nearest").float()
            reassembled = reassembled + shifted * weights[:, index : index + 1]
        base = F.interpolate(value, size=target_size, mode="nearest").float()
        return (reassembled - base).to(deep.dtype), weights, confidence

    def _energy_budget(self, base, correction):
        if not self.use_energy_budget:
            return correction, torch.ones((base.shape[0], base.shape[1], 1, 1), device=base.device, dtype=torch.float32)
        base_energy = base.float().square().mean(dim=(2, 3), keepdim=True).add(self.eps).sqrt()
        correction_energy = correction.float().square().mean(dim=(2, 3), keepdim=True).add(self.eps).sqrt()
        scale = torch.minimum(torch.ones_like(correction_energy), self.max_residual_ratio * base_energy / correction_energy.clamp_min(self.eps))
        if self.detach_budget:
            scale = scale.detach()
        return (correction.float() * scale).to(base.dtype), scale

    def compute_components(self, deep, lateral):
        self._validate_inputs(deep, lateral)
        base = F.interpolate(deep, size=lateral.shape[-2:], mode="nearest")
        residual, weights, confidence = self._reassemble(deep, self._build_query(lateral))
        if self.use_entropy:
            confidence_used = confidence.detach() if self.detach_confidence else confidence
            confidence_used = self.confidence_floor + (1.0 - self.confidence_floor) * confidence_used.float()
        else:
            confidence_used = torch.ones_like(confidence, dtype=torch.float32)
        correction = self.residual_out((residual.float() * confidence_used).to(residual.dtype))
        correction, budget_scale = self._energy_budget(base, correction)
        return base, correction, weights, confidence, budget_scale

    def forward(self, x):
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise TypeError("BCRAUp expects [P5_deep, P4_lateral].")
        base, correction, _, _, _ = self.compute_components(x[0], x[1])
        return base + correction


class MCASUp(nn.Module):
    """Multi-Basis Context-Adaptive Semantic Upsampling with an immutable nearest main path."""

    def __init__(
        self,
        c_deep,
        c_lateral,
        scale=2,
        reduction=4,
        temperature=1.0,
        residual_groups=4,
        max_residual_ratio=0.10,
        strict_scale=True,
        detach_budget=True,
        eps=1e-6,
    ):
        super().__init__()
        self.c_deep, self.c_lateral = int(c_deep), int(c_lateral)
        self.scale, self.temperature = int(scale), float(temperature)
        self.residual_groups_requested = int(residual_groups)
        self.max_residual_ratio = float(max_residual_ratio)
        self.strict_scale, self.detach_budget, self.eps = bool(strict_scale), bool(detach_budget), float(eps)
        if self.c_deep <= 0 or self.c_lateral <= 0:
            raise ValueError("MCASUp channel counts must be positive.")
        if self.scale <= 1 or int(reduction) < 1:
            raise ValueError("MCASUp requires scale > 1 and reduction >= 1.")
        if self.temperature <= 0.0 or not 0.0 < self.max_residual_ratio <= 1.0 or self.eps <= 0.0:
            raise ValueError("MCASUp requires positive temperature/eps and max_residual_ratio in (0, 1].")
        if self.residual_groups_requested <= 0:
            raise ValueError("MCASUp residual_groups must be positive.")

        self.hidden = max(16, min(64, min(self.c_deep, self.c_lateral) // int(reduction)))
        self.residual_groups = max(1, math.gcd(self.c_deep, self.residual_groups_requested))
        blur = torch.tensor(((1.0, 2.0, 1.0), (2.0, 4.0, 2.0), (1.0, 2.0, 1.0)), dtype=torch.float32)
        self.register_buffer("blur_kernel", (blur / blur.sum())[None, None], persistent=False)

        # New-layer construction must not perturb the RNG sequence used by downstream YOLO layers.
        with torch.random.fork_rng(devices=[], enabled=True):
            self.deep_proj = nn.Conv2d(self.c_deep, self.hidden, 1, bias=False)
            self.lateral_proj = nn.Conv2d(self.c_lateral, self.hidden, 1, bias=False)
            self.weight_predictor = nn.Sequential(
                nn.Conv2d(3 * self.hidden, self.hidden, 3, 1, 1, bias=True),
                nn.SiLU(inplace=True),
                nn.Conv2d(self.hidden, 3, 1, bias=True),
            )
            self.residual_out = nn.Conv2d(
                self.c_deep, self.c_deep, 1, groups=self.residual_groups, bias=False
            )
            nn.init.normal_(self.weight_predictor[-1].weight, mean=0.0, std=1e-3)
            with torch.no_grad():
                self.weight_predictor[-1].bias.copy_(self.weight_predictor[-1].bias.new_tensor((2.0, 0.0, -1.0)))
            nn.init.zeros_(self.residual_out.weight)

    def _validate_inputs(self, deep, lateral):
        if deep.ndim != 4 or lateral.ndim != 4:
            raise ValueError("MCASUp expects NCHW deep and lateral tensors.")
        if deep.shape[0] != lateral.shape[0]:
            raise ValueError("MCASUp deep and lateral batch sizes differ.")
        if deep.shape[1] != self.c_deep or lateral.shape[1] != self.c_lateral:
            raise ValueError(
                f"MCASUp expected deep/lateral channels {self.c_deep}/{self.c_lateral}, "
                f"got {deep.shape[1]}/{lateral.shape[1]}."
            )
        if deep.device != lateral.device or deep.dtype != lateral.dtype:
            raise ValueError("MCASUp deep and lateral tensors must share device and dtype.")
        expected = (deep.shape[-2] * self.scale, deep.shape[-1] * self.scale)
        if self.strict_scale and tuple(lateral.shape[-2:]) != expected:
            raise ValueError(f"MCASUp expected lateral size {expected}, got {tuple(lateral.shape[-2:])}.")

    def _blur(self, deep):
        # The validated fixed channel count keeps the grouped convolution exportable to ONNX.
        kernel = self.blur_kernel.to(device=deep.device, dtype=deep.dtype).repeat(self.c_deep, 1, 1, 1)
        return F.conv2d(F.pad(deep, (1, 1, 1, 1), mode="replicate"), kernel, groups=self.c_deep)

    def _energy_budget(self, base, correction):
        if base.shape != correction.shape:
            raise ValueError("MCASUp base and correction shapes differ.")
        # Do not inflate base energy with eps: a zero base must force an exactly zero correction.
        base_energy = base.float().square().mean(dim=(2, 3), keepdim=True).sqrt()
        correction_energy = correction.float().square().mean(dim=(2, 3), keepdim=True).add(self.eps).sqrt()
        scale = torch.minimum(
            torch.ones_like(correction_energy),
            self.max_residual_ratio * base_energy / correction_energy.clamp_min(self.eps),
        )
        if self.detach_budget:
            scale = scale.detach()
        return (correction.float() * scale).to(dtype=base.dtype), scale

    def compute_components(self, deep, lateral):
        """Return nearest base, bounded correction, convex basis weights, budget scale and raw basis residual."""
        self._validate_inputs(deep, lateral)
        target_size = lateral.shape[-2:]
        nearest_base = F.interpolate(deep, size=target_size, mode="nearest")
        bilinear_base = F.interpolate(deep, size=target_size, mode="bilinear", align_corners=False)
        smooth_base = F.interpolate(self._blur(deep), size=target_size, mode="bilinear", align_corners=False)

        deep_query = F.interpolate(self.deep_proj(deep), size=target_size, mode="bilinear", align_corners=False)
        lateral_query = self.lateral_proj(lateral)
        logits = self.weight_predictor(
            torch.cat((deep_query, lateral_query, (deep_query - lateral_query).abs()), dim=1)
        ).float() / self.temperature
        weights = logits.softmax(dim=1).to(dtype=nearest_base.dtype)
        mixed = (
            weights[:, 0:1] * nearest_base
            + weights[:, 1:2] * bilinear_base
            + weights[:, 2:3] * smooth_base
        )
        basis_residual = mixed - nearest_base
        correction, budget_scale = self._energy_budget(nearest_base, self.residual_out(basis_residual))
        return nearest_base, correction, weights, budget_scale, basis_residual

    def forward(self, x):
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise TypeError("MCASUp expects [P5_deep, P4_lateral].")
        nearest_base, correction, _, _, _ = self.compute_components(x[0], x[1])
        return nearest_base + correction


class ConsensusReliabilityRouter(nn.Module):
    """Route consensus-supported P2 evidence into P3 under a channel-release budget."""

    def __init__(
        self,
        c_shallow,
        c_out,
        max_gain=0.08,
        reduction=4,
        release_rho=0.25,
        unique_weight=0.35,
        local_self=0.75,
        eps=1e-6,
    ):
        super().__init__()
        self.c_shallow, self.c_out = int(c_shallow), int(c_out)
        self.max_gain, self.release_rho = float(max_gain), float(release_rho)
        self.unique_weight, self.local_self, self.eps = float(unique_weight), float(local_self), float(eps)
        reduction = int(reduction)
        if self.c_shallow <= 0 or self.c_out <= 0:
            raise ValueError("ConsensusReliabilityRouter channel counts must be positive.")
        if self.max_gain < 0.0 or reduction < 1 or self.eps <= 0.0:
            raise ValueError("max_gain must be non-negative; reduction and eps must be positive.")
        if not 0.0 <= self.release_rho <= 1.0:
            raise ValueError("release_rho must be in [0, 1].")
        if not 0.0 <= self.unique_weight <= 1.0 or not 0.0 <= self.local_self <= 1.0:
            raise ValueError("unique_weight and local_self must be in [0, 1].")

        hidden = max(self.c_out // reduction, 16)
        blur = torch.tensor(((1.0, 2.0, 1.0), (2.0, 4.0, 2.0), (1.0, 2.0, 1.0)), dtype=torch.float32)
        self.register_buffer("blur_kernel", (blur / blur.sum())[None, None], persistent=False)
        self.low_proj = Conv(self.c_shallow, self.c_out, 1, 1, act=False)
        self.detail_proj = Conv(self.c_shallow, self.c_out, 1, 1, act=False)
        self.route_gate = nn.Sequential(
            Conv(3 * self.c_out, hidden, 1, 1), nn.Conv2d(hidden, self.c_out, 1, bias=True), nn.Sigmoid()
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(4, 8, 3, 1, 1, bias=True), nn.SiLU(inplace=True), nn.Conv2d(8, 1, 1, bias=True), nn.Sigmoid()
        )
        self.channel_gate = nn.Sequential(
            nn.Conv2d(3 * self.c_out, hidden, 1, bias=True), nn.SiLU(inplace=True),
            nn.Conv2d(hidden, self.c_out, 1, bias=True), nn.Sigmoid(),
        )
        self.reliability_head = nn.Sequential(
            nn.Conv2d(4, 8, 3, 1, 1, bias=True), nn.SiLU(inplace=True), nn.Conv2d(8, 1, 1, bias=True), nn.Sigmoid()
        )
        self.out_proj = nn.Conv2d(self.c_out, self.c_out, 1, bias=True)
        nn.init.kaiming_normal_(self.out_proj.weight, mode="fan_out", nonlinearity="linear")
        nn.init.zeros_(self.out_proj.bias)
        self.alpha_raw = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        self.record_diagnostics = False
        self.latest_diagnostics = {}

    def set_diagnostics(self, enabled=True):
        self.record_diagnostics = bool(enabled)
        if not self.record_diagnostics:
            self.latest_diagnostics = {}
        return self

    def _blur_downsample(self, x):
        kernel = self.blur_kernel.to(device=x.device, dtype=x.dtype).repeat(self.c_shallow, 1, 1, 1)
        return F.conv2d(F.pad(x, (1, 1, 1, 1), mode="replicate"), kernel, stride=2, groups=self.c_shallow)

    @staticmethod
    def _align(x, size):
        return x if x.shape[-2:] == size else F.interpolate(x, size=size, mode="bilinear", align_corners=False)

    @staticmethod
    def _local_mean(x):
        return F.avg_pool2d(F.pad(x, (1, 1, 1, 1), mode="replicate"), 3, 1)

    @staticmethod
    def _spatial_zscore(x, eps):
        value = x.float()
        return (value - value.mean(dim=(2, 3), keepdim=True)) / value.var(dim=(2, 3), keepdim=True, unbiased=False).add(eps).sqrt()

    def _relative_support(self, detail_energy, low_energy):
        ratio = (detail_energy.float() / (low_energy.float() + self.eps)).clamp(0.0, 8.0)
        return ((1.0 - torch.exp(-ratio)) * torch.sigmoid(self._spatial_zscore(detail_energy, self.eps))).clamp(0.0, 1.0)

    def _channel_budget_gate(self, low, detail, semantic_detail):
        low_token = low.float().abs().mean(dim=(2, 3), keepdim=True)
        detail_token = detail.float().abs().mean(dim=(2, 3), keepdim=True)
        semantic_token = semantic_detail.float().abs().mean(dim=(2, 3), keepdim=True)

        def normalize(token):
            return token / (token.mean(dim=1, keepdim=True) + self.eps)

        detail_share = detail_token / (detail_token + semantic_token + self.eps)
        shallow_unique = (2.0 * detail_share - 1.0).clamp_min(0.0)
        coactivated = (1.0 - (2.0 * detail_share - 1.0).abs()).clamp(0.0, 1.0)
        channel_prior = (self.unique_weight * shallow_unique + coactivated).clamp(0.0, 1.0)
        tokens = torch.cat((normalize(low_token), normalize(detail_token), normalize(semantic_token)), dim=1).detach().to(low.dtype)
        raw_gate = self.channel_gate(tokens).float() * channel_prior.detach()
        raw_mean = raw_gate.mean(dim=1, keepdim=True)
        budget_scale = torch.minimum(torch.ones_like(raw_mean), torch.full_like(raw_mean, self.release_rho) / raw_mean.clamp_min(self.eps)).detach()
        return (raw_gate * budget_scale).clamp(0.0, 1.0).to(low.dtype)

    def forward(self, shallow, semantic):
        if shallow.ndim != 4 or semantic.ndim != 4 or shallow.shape[0] != semantic.shape[0]:
            raise ValueError("ConsensusReliabilityRouter expects compatible NCHW tensors.")
        if shallow.shape[1] != self.c_shallow or semantic.shape[1] != self.c_out:
            raise ValueError("ConsensusReliabilityRouter received unexpected channel counts.")
        target_size = semantic.shape[-2:]
        low_raw = self._align(self._blur_downsample(shallow), target_size)
        detail_raw = (F.adaptive_max_pool2d(shallow, target_size) - F.adaptive_avg_pool2d(shallow, target_size)).clamp_min(0.0)
        low, detail = self.low_proj(low_raw), self.detail_proj(detail_raw)
        semantic_float = semantic.float()
        semantic_low = self._local_mean(semantic_float)
        semantic_detail = (semantic_float - semantic_low).abs()
        shallow_support = self._relative_support(detail_raw.float().abs().mean(1, keepdim=True), low_raw.float().abs().mean(1, keepdim=True))
        semantic_support = self._relative_support(semantic_detail.mean(1, keepdim=True), semantic_low.abs().mean(1, keepdim=True))
        consensus, shallow_unique = shallow_support * semantic_support, shallow_support * (1.0 - semantic_support)
        spatial_prior = (consensus + self.unique_weight * shallow_unique).clamp(0.0, 1.0)
        spatial_prior = (self.local_self * spatial_prior + (1.0 - self.local_self) * self._local_mean(spatial_prior)).clamp(0.0, 1.0)
        cues = torch.cat((shallow_support, semantic_support, consensus, shallow_unique), dim=1).detach().to(semantic.dtype)
        spatial_release = spatial_prior.detach().to(semantic.dtype) * self.spatial_gate(cues)
        channel_release = self._channel_budget_gate(low, detail, semantic_detail.to(semantic.dtype))
        delta = self.out_proj(self.route_gate(torch.cat((low, detail, semantic), dim=1)) * (low + detail))
        gain = (self.max_gain * torch.tanh(self.alpha_raw)).to(semantic.dtype)
        routed = semantic + gain * spatial_release * channel_release * delta
        reliability = (0.50 * consensus.to(semantic.dtype) + 0.25 * semantic_support.to(semantic.dtype) + 0.25 * self.reliability_head(cues)).clamp(0.0, 1.0)
        if self.record_diagnostics:
            self.latest_diagnostics = {
                "gain": gain.detach(), "spatial_release": spatial_release.detach(),
                "channel_release": channel_release.detach(), "reliability": reliability.detach(), "consensus": consensus.detach(),
            }
        return routed, reliability


class SymmetricMomentPreservedGeometry(nn.Module):
    """Five-point contour geometry whose paired supports make the first moment exactly zero."""

    def __init__(
        self,
        channels,
        geom_ratio=0.50,
        min_radius=0.25,
        max_radius=1.25,
        center_floor=0.50,
        center_ceiling=0.80,
        max_angle_deg=22.5,
        max_gain=0.08,
        reduction=4,
        eps=1e-6,
    ):
        super().__init__()
        self.channels, self.min_radius, self.max_radius = int(channels), float(min_radius), float(max_radius)
        self.center_floor, self.center_ceiling = float(center_floor), float(center_ceiling)
        self.max_angle, self.max_gain, self.eps = math.radians(float(max_angle_deg)), float(max_gain), float(eps)
        reduction = int(reduction)
        if self.channels <= 0 or not 0.0 < float(geom_ratio) <= 1.0:
            raise ValueError("channels must be positive and geom_ratio must be in (0, 1].")
        if not 0.0 <= self.min_radius <= self.max_radius or not 0.0 <= self.center_floor <= self.center_ceiling <= 1.0:
            raise ValueError("invalid radius or center-mass bounds.")
        if not 0.0 <= self.max_angle <= math.pi / 2 or self.max_gain < 0.0 or reduction < 1 or self.eps <= 0.0:
            raise ValueError("invalid geometry parameters.")
        raw_channels = int(round(self.channels * float(geom_ratio) / 8.0)) * 8
        self.geom_channels = min(self.channels, max(raw_channels, 8))
        hidden = max(self.geom_channels // reduction, 16)
        self.reduce = Conv(self.channels, self.geom_channels, 1, 1)
        self.geometry_trunk = nn.Sequential(
            Conv(self.geom_channels + 3, hidden, 3, 1), Conv(hidden, hidden, 3, 1, g=hidden), Conv(hidden, hidden, 1, 1)
        )
        self.radius_head = nn.Conv2d(hidden, 2, 1, bias=True)
        self.pair_mass_head = nn.Conv2d(hidden, 2, 1, bias=True)
        self.center_head = nn.Conv2d(hidden, 1, 1, bias=True)
        self.angle_head = nn.Conv2d(hidden, 1, 1, bias=True)
        for layer in (self.radius_head, self.pair_mass_head, self.center_head, self.angle_head):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        self.out_proj = nn.Conv2d(self.geom_channels, self.channels, 1, bias=True)
        nn.init.kaiming_normal_(self.out_proj.weight, mode="fan_out", nonlinearity="linear")
        nn.init.zeros_(self.out_proj.bias)
        sobel_x = torch.tensor(((-1.0, 0.0, 1.0), (-2.0, 0.0, 2.0), (-1.0, 0.0, 1.0)), dtype=torch.float32)
        self.register_buffer("sobel_x", sobel_x[None, None], persistent=False)
        self.register_buffer("sobel_y", sobel_x.t().contiguous()[None, None], persistent=False)
        self.alpha_raw = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        self.record_diagnostics = False
        self.latest_diagnostics = {}

    def set_diagnostics(self, enabled=True):
        self.record_diagnostics = bool(enabled)
        if not self.record_diagnostics:
            self.latest_diagnostics = {}
        return self

    def _contour_cues(self, x):
        summary = F.pad(x.float().mean(dim=1, keepdim=True), (1, 1, 1, 1), mode="replicate")
        grad_x_raw = F.conv2d(summary, self.sobel_x.to(device=x.device, dtype=summary.dtype))
        grad_y_raw = F.conv2d(summary, self.sobel_y.to(device=x.device, dtype=summary.dtype))
        global_rms = (grad_x_raw.square() + grad_y_raw.square()).mean(dim=(2, 3), keepdim=True).add(self.eps).sqrt()
        grad_x, grad_y = grad_x_raw / global_rms, grad_y_raw / global_rms
        magnitude = (grad_x.square() + grad_y.square() + self.eps).sqrt()
        return grad_x.to(x.dtype), grad_y.to(x.dtype), magnitude.to(x.dtype)

    @staticmethod
    def _base_grid(batch, height, width, device):
        y = (torch.arange(height, device=device, dtype=torch.float32) + 0.5) * (2.0 / max(height, 1)) - 1.0
        x = (torch.arange(width, device=device, dtype=torch.float32) + 0.5) * (2.0 / max(width, 1)) - 1.0
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        return torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0).expand(batch, -1, -1, -1)

    def _predict_support(self, feature, reliability):
        grad_x, grad_y, magnitude = self._contour_cues(feature)
        geometry = self.geometry_trunk(torch.cat((feature, grad_x, grad_y, magnitude), dim=1))
        batch, _, height, width = feature.shape
        if reliability.shape[-2:] != (height, width):
            reliability = F.interpolate(reliability, size=(height, width), mode="bilinear", align_corners=False)
        if reliability.shape[0] != batch or reliability.shape[1] != 1:
            raise ValueError("reliability must have shape [B, 1, H, W].")
        reliability = reliability.float().clamp(0.0, 1.0)
        local_norm = (grad_x.float().square() + grad_y.float().square() + self.eps).sqrt()
        normal_x, normal_y = grad_x.float() / local_norm, grad_y.float() / local_norm
        tangent_x, tangent_y = -normal_y, normal_x
        orientation_confidence = (magnitude.float() / (magnitude.float() + 1.0)).clamp(0.0, 1.0)
        delta_angle = self.max_angle * torch.tanh(self.angle_head(geometry).float()) * orientation_confidence
        cos_a, sin_a = torch.cos(delta_angle), torch.sin(delta_angle)
        rotated_tangent_x, rotated_tangent_y = cos_a * tangent_x - sin_a * tangent_y, sin_a * tangent_x + cos_a * tangent_y
        rotated_normal_x, rotated_normal_y = -rotated_tangent_y, rotated_tangent_x
        base_radius = self.min_radius + (self.max_radius - self.min_radius) * (1.0 - reliability)
        radius_scale = 0.50 + 0.50 * torch.sigmoid(self.radius_head(geometry).float())
        tangent_radius = (base_radius * radius_scale[:, 0:1]).clamp(self.min_radius, self.max_radius)
        normal_radius = (base_radius * radius_scale[:, 1:2]).clamp(self.min_radius, self.max_radius)
        zero = torch.zeros_like(tangent_radius)
        tangent_offset = torch.cat((tangent_radius * rotated_tangent_x, tangent_radius * rotated_tangent_y), dim=1)
        normal_offset = torch.cat((normal_radius * rotated_normal_x, normal_radius * rotated_normal_y), dim=1)
        offsets = torch.stack((torch.cat((zero, zero), dim=1), tangent_offset, -tangent_offset, normal_offset, -normal_offset), dim=1).to(feature.dtype)
        learned_center = torch.sigmoid(self.center_head(geometry).float())
        center_mass = self.center_floor + (self.center_ceiling - self.center_floor) * (0.65 * reliability + 0.35 * learned_center).clamp(0.0, 1.0)
        pair_mix = self.pair_mass_head(geometry).float().softmax(dim=1)
        remaining = 1.0 - center_mass
        tangent_pair_mass, normal_pair_mass = remaining * pair_mix[:, 0:1], remaining * pair_mix[:, 1:2]
        weights = torch.cat((center_mass, 0.5 * tangent_pair_mass, 0.5 * tangent_pair_mass, 0.5 * normal_pair_mass, 0.5 * normal_pair_mass), dim=1).to(feature.dtype)
        return offsets, weights, magnitude, reliability

    def _sample(self, x, offsets):
        batch, channels, height, width = x.shape
        base = self._base_grid(batch, height, width, x.device).unsqueeze(1)
        offset_x, offset_y = offsets[:, :, 0].float() * (2.0 / max(width, 1)), offsets[:, :, 1].float() * (2.0 / max(height, 1))
        grid = base + torch.stack((offset_x, offset_y), dim=-1)
        source = x.float().unsqueeze(1).expand(-1, 5, -1, -1, -1).reshape(batch * 5, channels, height, width)
        sampled = F.grid_sample(source, grid.reshape(batch * 5, height, width, 2), mode="bilinear", padding_mode="border", align_corners=False)
        return sampled.reshape(batch, 5, channels, height, width).to(x.dtype)

    def forward(self, x, reliability):
        if x.ndim != 4 or reliability.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError("SymmetricMomentPreservedGeometry expects x=[B,C,H,W] and reliability=[B,1,H,W].")
        if reliability.shape[0] != x.shape[0] or reliability.shape[1] != 1:
            raise ValueError("reliability must have shape [B,1,H,W].")
        geometry_feature = self.reduce(x)
        offsets, weights, magnitude, reliability_used = self._predict_support(geometry_feature, reliability)
        aggregated = (self._sample(geometry_feature, offsets) * weights.unsqueeze(2).to(geometry_feature.dtype)).sum(dim=1)
        gain = (self.max_gain * torch.tanh(self.alpha_raw)).to(x.dtype)
        output = x + gain * self.out_proj(aggregated - geometry_feature)
        if self.record_diagnostics:
            self.latest_diagnostics = {
                "gain": gain.detach(), "weights": weights.detach(), "offsets": offsets.detach(),
                "first_moment": (offsets.float() * weights.unsqueeze(2).float()).sum(dim=1).detach(),
                "magnitude": magnitude.detach(), "reliability": reliability_used.detach(),
            }
        return output

class CMRFDSC3k2(DSC3k2):
    """DSC3k2 main path enhanced by CRR and a reliability-controlled SMPG branch."""

    def __init__(
        self, c_shallow, c_p3, c2, n=1, dsc3k=False, e=0.25, geom_ratio=0.50, min_radius=0.25,
        max_radius=1.25, center_floor=0.50, center_ceiling=0.80, max_angle_deg=22.5, detail_gain=0.08,
        geom_gain=0.08, reduction=4, release_rho=0.25, unique_weight=0.35, local_self=0.75,
        g=1, shortcut=True, k1=3, k2=7, d2=1,
    ):
        super().__init__(c1=c_p3, c2=c2, n=n, dsc3k=dsc3k, e=e, g=g, shortcut=shortcut, k1=k1, k2=k2, d2=d2)
        self.c_shallow, self.c_p3, self.c2 = int(c_shallow), int(c_p3), int(c2)
        self.dsc3k_enabled = bool(dsc3k)
        with torch.random.fork_rng(devices=[], enabled=True):
            self.evidence_router = ConsensusReliabilityRouter(
                self.c_shallow, self.c2, max_gain=detail_gain, reduction=reduction, release_rho=release_rho,
                unique_weight=unique_weight, local_self=local_self,
            )
            self.geometry = SymmetricMomentPreservedGeometry(
                self.c2, geom_ratio=geom_ratio, min_radius=min_radius, max_radius=max_radius,
                center_floor=center_floor, center_ceiling=center_ceiling, max_angle_deg=max_angle_deg,
                max_gain=geom_gain, reduction=reduction,
            )

    def forward(self, x):
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise ValueError("CMRFDSC3k2 expects [P2_shallow, P3_downsampled].")
        shallow, p3 = x
        if shallow.ndim != 4 or p3.ndim != 4 or shallow.shape[0] != p3.shape[0]:
            raise ValueError("CMRFDSC3k2 inputs must be compatible NCHW tensors.")
        if shallow.shape[1] != self.c_shallow or p3.shape[1] != self.c_p3:
            raise ValueError("CMRFDSC3k2 received unexpected P2/P3 channel counts.")
        routed, reliability = self.evidence_router(shallow, super().forward(p3))
        return self.geometry(routed, reliability)

"""CARM-YOLOv13 structural modules.

Place these classes in ultralytics/nn/modules/block.py after CMRFDSC3k2.
Required existing symbols: Conv, DSC3k2, CMRFDSC3k2, torch, nn, F, math.
"""


class MicroObjectMomentRefiner(nn.Module):
    """Refine compact P3 structures with bounded, symmetric five-point sampling.

    The sampling supports are [center, +tangent, -tangent, +normal, -normal].
    Paired weights are shared, so the weighted first offset moment is exactly zero.
    A shallow-detail prior and a spatial release budget restrict the branch to compact
    candidate regions. The zero-start gain makes the module an exact identity at init.
    """

    def __init__(
        self,
        c_shallow,
        channels,
        geom_ratio=0.25,
        min_radius=0.20,
        max_radius=0.75,
        center_floor=0.65,
        center_ceiling=0.90,
        max_gain=0.06,
        spatial_rho=0.15,
        reduction=4,
        eps=1e-6,
    ):
        super().__init__()
        self.c_shallow = int(c_shallow)
        self.channels = int(channels)
        self.min_radius = float(min_radius)
        self.max_radius = float(max_radius)
        self.center_floor = float(center_floor)
        self.center_ceiling = float(center_ceiling)
        self.max_gain = float(max_gain)
        self.spatial_rho = float(spatial_rho)
        self.eps = float(eps)
        reduction = int(reduction)

        if self.c_shallow <= 0 or self.channels <= 0:
            raise ValueError("MicroObjectMomentRefiner channel counts must be positive.")
        if not 0.0 < float(geom_ratio) <= 1.0:
            raise ValueError("geom_ratio must be in (0, 1].")
        if not 0.0 <= self.min_radius <= self.max_radius:
            raise ValueError("Require 0 <= min_radius <= max_radius.")
        if not 0.0 <= self.center_floor <= self.center_ceiling <= 1.0:
            raise ValueError("Require 0 <= center_floor <= center_ceiling <= 1.")
        if self.max_gain < 0.0 or not 0.0 <= self.spatial_rho <= 1.0:
            raise ValueError("max_gain must be non-negative and spatial_rho in [0, 1].")
        if reduction < 1 or self.eps <= 0.0:
            raise ValueError("reduction and eps must be positive.")

        raw_channels = int(round(self.channels * float(geom_ratio) / 8.0)) * 8
        self.geom_channels = min(self.channels, max(raw_channels, 8))
        hidden = max(self.geom_channels // reduction, 16)

        blur = torch.tensor(
            ((1.0, 2.0, 1.0), (2.0, 4.0, 2.0), (1.0, 2.0, 1.0)),
            dtype=torch.float32,
        )
        self.register_buffer("blur_kernel", (blur / blur.sum())[None, None], persistent=False)
        sobel_x = torch.tensor(
            ((-1.0, 0.0, 1.0), (-2.0, 0.0, 2.0), (-1.0, 0.0, 1.0)),
            dtype=torch.float32,
        )
        self.register_buffer("sobel_x", sobel_x[None, None], persistent=False)
        self.register_buffer("sobel_y", sobel_x.t().contiguous()[None, None], persistent=False)

        self.reduce = Conv(self.channels, self.geom_channels, 1, 1)
        self.shallow_proj = Conv(self.c_shallow, self.geom_channels, 1, 1, act=False)
        self.prior_head = nn.Sequential(
            nn.Conv2d(4, 8, 3, 1, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(8, 1, 1, bias=True),
            nn.Sigmoid(),
        )
        self.geometry_trunk = nn.Sequential(
            Conv(2 * self.geom_channels + 3, hidden, 3, 1),
            Conv(hidden, hidden, 3, 1, g=hidden),
            Conv(hidden, hidden, 1, 1),
        )
        self.radius_head = nn.Conv2d(hidden, 2, 1, bias=True)
        self.center_head = nn.Conv2d(hidden, 1, 1, bias=True)
        self.pair_mass_head = nn.Conv2d(hidden, 2, 1, bias=True)
        for layer in (self.radius_head, self.center_head, self.pair_mass_head):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

        self.out_proj = nn.Conv2d(self.geom_channels, self.channels, 1, bias=True)
        nn.init.kaiming_normal_(self.out_proj.weight, mode="fan_out", nonlinearity="linear")
        nn.init.zeros_(self.out_proj.bias)
        nn.init.normal_(self.prior_head[-2].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.prior_head[-2].bias)

        self.alpha_raw = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        self.record_diagnostics = False
        self.latest_diagnostics = {}

    def set_diagnostics(self, enabled=True):
        self.record_diagnostics = bool(enabled)
        if not self.record_diagnostics:
            self.latest_diagnostics = {}
        return self

    @staticmethod
    def _local_mean(x, kernel_size=3):
        radius = kernel_size // 2
        return F.avg_pool2d(
            F.pad(x, (radius, radius, radius, radius), mode="replicate"),
            kernel_size,
            1,
        )

    def _blur_downsample(self, x):
        kernel = self.blur_kernel.to(device=x.device, dtype=x.dtype).repeat(self.c_shallow, 1, 1, 1)
        return F.conv2d(
            F.pad(x, (1, 1, 1, 1), mode="replicate"),
            kernel,
            stride=2,
            groups=self.c_shallow,
        )

    @staticmethod
    def _align(x, size):
        return x if x.shape[-2:] == size else F.interpolate(x, size=size, mode="bilinear", align_corners=False)

    @staticmethod
    def _spatial_zscore(x, eps):
        value = x.float()
        mean = value.mean(dim=(2, 3), keepdim=True)
        std = value.var(dim=(2, 3), keepdim=True, unbiased=False).add(eps).sqrt()
        return (value - mean) / std

    def _relative_support(self, detail_energy, low_energy):
        ratio = (detail_energy.float() / (low_energy.float() + self.eps)).clamp(0.0, 8.0)
        absolute = 1.0 - torch.exp(-ratio)
        relative = torch.sigmoid(self._spatial_zscore(detail_energy, self.eps))
        return (absolute * relative).clamp(0.0, 1.0)

    def _spatial_budget(self, gate):
        mean = gate.float().mean(dim=(2, 3), keepdim=True)
        scale = torch.minimum(
            torch.ones_like(mean),
            torch.full_like(mean, self.spatial_rho) / mean.clamp_min(self.eps),
        ).detach()
        return (gate.float() * scale).clamp(0.0, 1.0).to(gate.dtype), scale

    def _contour_cues(self, x):
        summary = F.pad(x.float().mean(dim=1, keepdim=True), (1, 1, 1, 1), mode="replicate")
        grad_x_raw = F.conv2d(summary, self.sobel_x.to(device=x.device, dtype=summary.dtype))
        grad_y_raw = F.conv2d(summary, self.sobel_y.to(device=x.device, dtype=summary.dtype))
        rms = (grad_x_raw.square() + grad_y_raw.square()).mean(dim=(2, 3), keepdim=True).add(self.eps).sqrt()
        grad_x = grad_x_raw / rms
        grad_y = grad_y_raw / rms
        magnitude = (grad_x.square() + grad_y.square() + self.eps).sqrt()
        return grad_x.to(x.dtype), grad_y.to(x.dtype), magnitude.to(x.dtype)

    @staticmethod
    def _base_grid(batch, height, width, device):
        y = (torch.arange(height, device=device, dtype=torch.float32) + 0.5) * (2.0 / max(height, 1)) - 1.0
        x = (torch.arange(width, device=device, dtype=torch.float32) + 0.5) * (2.0 / max(width, 1)) - 1.0
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        return torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0).expand(batch, -1, -1, -1)

    def _sample(self, x, offsets):
        batch, channels, height, width = x.shape
        if offsets.shape != (batch, 5, 2, height, width):
            raise ValueError(f"Unexpected offset shape {tuple(offsets.shape)}.")
        base = self._base_grid(batch, height, width, x.device).unsqueeze(1)
        offset_x = offsets[:, :, 0].float() * (2.0 / max(width, 1))
        offset_y = offsets[:, :, 1].float() * (2.0 / max(height, 1))
        grid = base + torch.stack((offset_x, offset_y), dim=-1)
        source = x.float().unsqueeze(1).expand(-1, 5, -1, -1, -1).reshape(
            batch * 5, channels, height, width
        )
        sampled = F.grid_sample(
            source,
            grid.reshape(batch * 5, height, width, 2),
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        return sampled.reshape(batch, 5, channels, height, width).to(dtype=x.dtype)

    def compute_components(self, shallow, semantic):
        if shallow.ndim != 4 or semantic.ndim != 4 or shallow.shape[0] != semantic.shape[0]:
            raise ValueError("MicroObjectMomentRefiner expects compatible NCHW tensors.")
        if shallow.shape[1] != self.c_shallow or semantic.shape[1] != self.channels:
            raise ValueError("MicroObjectMomentRefiner received unexpected channel counts.")
        if shallow.device != semantic.device or shallow.dtype != semantic.dtype:
            raise ValueError("MicroObjectMomentRefiner inputs must share device and dtype.")

        target_size = semantic.shape[-2:]
        shallow_low_raw = self._align(self._blur_downsample(shallow), target_size)
        shallow_avg_raw = F.adaptive_avg_pool2d(shallow, target_size)
        shallow_detail_raw = (F.adaptive_max_pool2d(shallow, target_size) - shallow_avg_raw).clamp_min(0.0)

        shallow_detail_energy = shallow_detail_raw.float().abs().mean(dim=1, keepdim=True)
        shallow_low_energy = shallow_low_raw.float().abs().mean(dim=1, keepdim=True)
        shallow_support = self._relative_support(shallow_detail_energy, shallow_low_energy)

        semantic_float = semantic.float()
        semantic_low = self._local_mean(semantic_float, 3)
        semantic_detail = (semantic_float - semantic_low).abs()
        semantic_support = self._relative_support(
            semantic_detail.mean(dim=1, keepdim=True),
            semantic_low.abs().mean(dim=1, keepdim=True),
        )

        local_detail = self._local_mean(shallow_detail_energy, 5)
        compact_ratio = (shallow_detail_energy / (local_detail + self.eps)).clamp(0.0, 8.0)
        compactness = (1.0 - torch.exp(-compact_ratio)).clamp(0.0, 1.0)
        analytic_prior = (shallow_support * compactness * (0.35 + 0.65 * semantic_support)).clamp(0.0, 1.0)
        prior_cues = torch.cat(
            (shallow_support, semantic_support, compactness, analytic_prior), dim=1
        ).detach().to(dtype=semantic.dtype)
        learned_prior = self.prior_head(prior_cues)
        micro_gate, budget_scale = self._spatial_budget(
            analytic_prior.detach().to(dtype=semantic.dtype) * learned_prior
        )

        feature = self.reduce(semantic)
        shallow_feature = self.shallow_proj(shallow_detail_raw)
        grad_x, grad_y, magnitude = self._contour_cues(feature)
        geometry = self.geometry_trunk(
            torch.cat((feature, shallow_feature, grad_x, grad_y, magnitude), dim=1)
        )

        local_norm = (grad_x.float().square() + grad_y.float().square() + self.eps).sqrt()
        normal_x = grad_x.float() / local_norm
        normal_y = grad_y.float() / local_norm
        tangent_x = -normal_y
        tangent_y = normal_x

        base_radius = self.min_radius + (self.max_radius - self.min_radius) * (1.0 - micro_gate.float())
        radius_scale = 0.75 + 0.25 * torch.sigmoid(self.radius_head(geometry).float())
        tangent_radius = (base_radius * radius_scale[:, 0:1]).clamp(self.min_radius, self.max_radius)
        normal_radius = (base_radius * radius_scale[:, 1:2]).clamp(self.min_radius, self.max_radius)

        tangent_offset = torch.cat((tangent_radius * tangent_x, tangent_radius * tangent_y), dim=1)
        normal_offset = torch.cat((normal_radius * normal_x, normal_radius * normal_y), dim=1)
        zero_offset = torch.zeros_like(tangent_offset)
        offsets = torch.stack(
            (zero_offset, tangent_offset, -tangent_offset, normal_offset, -normal_offset), dim=1
        ).to(dtype=feature.dtype)

        learned_center = torch.sigmoid(self.center_head(geometry).float())
        center_mass = self.center_floor + (self.center_ceiling - self.center_floor) * (
            0.65 * micro_gate.float() + 0.35 * learned_center
        ).clamp(0.0, 1.0)
        pair_mix = self.pair_mass_head(geometry).float().softmax(dim=1)
        remaining = 1.0 - center_mass
        tangent_mass = remaining * pair_mix[:, 0:1]
        normal_mass = remaining * pair_mix[:, 1:2]
        weights = torch.cat(
            (
                center_mass,
                0.5 * tangent_mass,
                0.5 * tangent_mass,
                0.5 * normal_mass,
                0.5 * normal_mass,
            ),
            dim=1,
        ).to(dtype=feature.dtype)

        sampled = self._sample(feature, offsets)
        aggregated = (sampled * weights.unsqueeze(2)).sum(dim=1)
        residual = micro_gate * self.out_proj(aggregated - feature)
        return residual, micro_gate, offsets, weights, budget_scale

    def forward(self, shallow, semantic):
        residual, micro_gate, offsets, weights, budget_scale = self.compute_components(shallow, semantic)
        gain = (self.max_gain * torch.tanh(self.alpha_raw)).to(dtype=semantic.dtype)
        output = semantic + gain * residual
        if self.record_diagnostics:
            self.latest_diagnostics = {
                "gain": gain.detach(),
                "micro_gate": micro_gate.detach(),
                "offsets": offsets.detach(),
                "weights": weights.detach(),
                "first_moment": (offsets.float() * weights.unsqueeze(2).float()).sum(dim=1).detach(),
                "budget_scale": budget_scale.detach(),
            }
        return output


class CARMDSC3k2(CMRFDSC3k2):
    """CMRF C4 main path plus a zero-start micro-object moment refiner."""

    def __init__(
        self,
        c_shallow,
        c_p3,
        c2,
        n=1,
        dsc3k=False,
        e=0.25,
        geom_ratio=0.50,
        min_radius=0.25,
        max_radius=1.25,
        center_floor=0.50,
        center_ceiling=0.80,
        max_angle_deg=22.5,
        detail_gain=0.08,
        geom_gain=0.08,
        reduction=4,
        release_rho=0.25,
        unique_weight=0.35,
        local_self=0.75,
        micro_ratio=0.25,
        micro_min_radius=0.20,
        micro_max_radius=0.75,
        micro_center_floor=0.65,
        micro_center_ceiling=0.90,
        micro_gain=0.06,
        micro_spatial_rho=0.15,
        micro_reduction=4,
        g=1,
        shortcut=True,
        k1=3,
        k2=7,
        d2=1,
    ):
        super().__init__(
            c_shallow=c_shallow,
            c_p3=c_p3,
            c2=c2,
            n=n,
            dsc3k=dsc3k,
            e=e,
            geom_ratio=geom_ratio,
            min_radius=min_radius,
            max_radius=max_radius,
            center_floor=center_floor,
            center_ceiling=center_ceiling,
            max_angle_deg=max_angle_deg,
            detail_gain=detail_gain,
            geom_gain=geom_gain,
            reduction=reduction,
            release_rho=release_rho,
            unique_weight=unique_weight,
            local_self=local_self,
            g=g,
            shortcut=shortcut,
            k1=k1,
            k2=k2,
            d2=d2,
        )
        with torch.random.fork_rng(devices=[], enabled=True):
            self.micro_refiner = MicroObjectMomentRefiner(
                c_shallow=self.c_shallow,
                channels=self.c2,
                geom_ratio=micro_ratio,
                min_radius=micro_min_radius,
                max_radius=micro_max_radius,
                center_floor=micro_center_floor,
                center_ceiling=micro_center_ceiling,
                max_gain=micro_gain,
                spatial_rho=micro_spatial_rho,
                reduction=micro_reduction,
            )

    def forward(self, x):
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise ValueError("CARMDSC3k2 expects [P2_shallow, P3_downsampled].")
        shallow, _ = x
        return self.micro_refiner(shallow, super().forward(x))



class ReliabilityFrequencyAlignUp(nn.Module):
    """Nearest P4-to-P3 main path with a reliability-gated, RMS-bounded frequency residual."""

    def __init__(
        self, c_deep, c_lateral, scale=2, reduction=4, max_gain=0.12, max_residual_ratio=0.15,
        boundary_floor=0.20, residual_groups=4, strict_scale=True, detach_budget=True, eps=1e-6,
    ):
        super().__init__()
        self.c_deep, self.c_lateral, self.scale = int(c_deep), int(c_lateral), int(scale)
        self.max_gain, self.max_residual_ratio, self.boundary_floor = float(max_gain), float(max_residual_ratio), float(boundary_floor)
        self.strict_scale, self.detach_budget, self.eps = bool(strict_scale), bool(detach_budget), float(eps)
        reduction, residual_groups = int(reduction), int(residual_groups)
        if self.c_deep <= 0 or self.c_lateral <= 0 or self.scale <= 1 or reduction < 1 or residual_groups < 1:
            raise ValueError("invalid ReliabilityFrequencyAlignUp channel, scale, reduction, or group setting.")
        if self.max_gain < 0.0 or not 0.0 < self.max_residual_ratio <= 1.0 or not 0.0 <= self.boundary_floor <= 1.0 or self.eps <= 0.0:
            raise ValueError("invalid ReliabilityFrequencyAlignUp gain, budget, boundary floor, or eps.")
        self.hidden = max(16, min(64, min(self.c_deep, self.c_lateral) // reduction))
        self.residual_groups = max(1, math.gcd(math.gcd(self.hidden, self.c_deep), residual_groups))
        blur = torch.tensor(((1.0, 2.0, 1.0), (2.0, 4.0, 2.0), (1.0, 2.0, 1.0)), dtype=torch.float32)
        self.register_buffer("blur_kernel", (blur / blur.sum())[None, None], persistent=False)
        with torch.random.fork_rng(devices=[], enabled=True):
            self.deep_proj = nn.Conv2d(self.c_deep, self.hidden, 1, bias=False)
            self.lateral_proj = nn.Conv2d(self.c_lateral, self.hidden, 1, bias=False)
            self.mix_gate = nn.Sequential(
                nn.Conv2d(3 * self.hidden, self.hidden, 3, 1, 1, bias=True), nn.SiLU(inplace=True),
                nn.Conv2d(self.hidden, 2, 1, bias=True), nn.Sigmoid(),
            )
            self.residual_out = nn.Conv2d(self.hidden, self.c_deep, 1, groups=self.residual_groups, bias=True)
            nn.init.kaiming_normal_(self.residual_out.weight, mode="fan_out", nonlinearity="linear")
            nn.init.zeros_(self.residual_out.bias)
        self.alpha_raw = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        self.record_diagnostics = False
        self.latest_diagnostics = {}

    def set_diagnostics(self, enabled=True):
        self.record_diagnostics = bool(enabled)
        if not self.record_diagnostics:
            self.latest_diagnostics = {}
        return self

    def _validate_inputs(self, deep, lateral):
        if deep.ndim != 4 or lateral.ndim != 4 or deep.shape[0] != lateral.shape[0]:
            raise ValueError("ReliabilityFrequencyAlignUp expects compatible NCHW tensors.")
        if deep.shape[1] != self.c_deep or lateral.shape[1] != self.c_lateral:
            raise ValueError("ReliabilityFrequencyAlignUp received unexpected channel counts.")
        if deep.device != lateral.device or deep.dtype != lateral.dtype:
            raise ValueError("ReliabilityFrequencyAlignUp inputs must share device and dtype.")
        expected = (deep.shape[-2] * self.scale, deep.shape[-1] * self.scale)
        if self.strict_scale and tuple(lateral.shape[-2:]) != expected:
            raise ValueError(f"expected lateral size={expected}, got {tuple(lateral.shape[-2:])}.")

    def _blur(self, x):
        kernel = self.blur_kernel.to(device=x.device, dtype=x.dtype).repeat(self.hidden, 1, 1, 1)
        return F.conv2d(F.pad(x, (1, 1, 1, 1), mode="replicate"), kernel, groups=self.hidden)

    def _energy_budget(self, base, correction):
        base_rms = base.float().square().mean(dim=(2, 3), keepdim=True).sqrt()
        correction_rms = correction.float().square().mean(dim=(2, 3), keepdim=True).sqrt()
        scale = torch.minimum(torch.ones_like(correction_rms), self.max_residual_ratio * base_rms / (correction_rms + self.eps))
        if self.detach_budget:
            scale = scale.detach()
        return (correction.float() * scale).to(base.dtype), scale

    def compute_components(self, deep, lateral):
        self._validate_inputs(deep, lateral)
        target_size = lateral.shape[-2:]
        base = F.interpolate(deep, size=target_size, mode="nearest")
        deep_embed = F.interpolate(self.deep_proj(deep), size=target_size, mode="nearest")
        lateral_embed = self.lateral_proj(lateral)
        deep_low, lateral_low = self._blur(deep_embed), self._blur(lateral_embed)
        lateral_high, low_residual = lateral_embed - lateral_low, lateral_low - deep_low
        cosine = ((F.normalize(deep_low.float(), dim=1, eps=self.eps) * F.normalize(lateral_low.float(), dim=1, eps=self.eps)).sum(1, keepdim=True) + 1.0).mul(0.5).clamp(0.0, 1.0)
        discrepancy = low_residual.float().abs().mean(1, keepdim=True)
        reference = deep_low.float().abs().mean(1, keepdim=True) + lateral_low.float().abs().mean(1, keepdim=True) + self.eps
        reliability = (0.5 * cosine + 0.5 * torch.exp(-(discrepancy / reference).clamp(0.0, 8.0))).clamp(0.0, 1.0)
        high_energy = lateral_high.float().abs().mean(1, keepdim=True)
        high_std = high_energy.var(dim=(2, 3), keepdim=True, unbiased=False).add(self.eps).sqrt()
        boundary_support = torch.sigmoid((high_energy - high_energy.mean(dim=(2, 3), keepdim=True)) / high_std)
        boundary_confidence = self.boundary_floor + (1.0 - self.boundary_floor) * boundary_support
        gates = self.mix_gate(torch.cat((deep_embed, lateral_embed, (deep_embed - lateral_embed).abs()), dim=1)).float()
        mixed_hidden = (gates[:, 0:1] * reliability * low_residual.float() + gates[:, 1:2] * boundary_confidence * lateral_high.float()).to(deep.dtype)
        correction, budget_scale = self._energy_budget(base, self.residual_out(mixed_hidden))
        return base, correction, reliability, boundary_support, gates, budget_scale

    def forward(self, x):
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise TypeError("ReliabilityFrequencyAlignUp expects [deep_feature, lateral_feature].")
        base, correction, reliability, boundary_support, gates, budget_scale = self.compute_components(x[0], x[1])
        gain = (self.max_gain * torch.tanh(self.alpha_raw)).to(base.dtype)
        output = base + gain * correction
        if self.record_diagnostics:
            self.latest_diagnostics = {
                "gain": gain.detach(), "reliability": reliability.detach(), "boundary_support": boundary_support.detach(),
                "gates": gates.detach(), "budget_scale": budget_scale.detach(),
            }
        return output

class OrthogonalComplementaryAlignUp(nn.Module):
    """Precision-safe P4-to-P3 upsampling with an orthogonal high-frequency residual.

    The main path is immutable nearest-neighbor upsampling. Low-frequency cross-scale
    agreement only gates a lateral high-frequency candidate. Before injection, the
    correction is local-mean removed, made channel-wise orthogonal to the nearest base,
    and bounded by a sample-wise scalar RMS budget. The branch is an exact identity at init.
    """

    def __init__(
        self,
        c_deep,
        c_lateral,
        scale=2,
        reduction=4,
        max_gain=0.12,
        max_residual_ratio=0.08,
        candidate_floor=0.20,
        residual_groups=4,
        strict_scale=True,
        detach_budget=True,
        eps=1e-6,
    ):
        super().__init__()
        self.c_deep = int(c_deep)
        self.c_lateral = int(c_lateral)
        self.scale = int(scale)
        self.max_gain = float(max_gain)
        self.max_residual_ratio = float(max_residual_ratio)
        self.candidate_floor = float(candidate_floor)
        self.strict_scale = bool(strict_scale)
        self.detach_budget = bool(detach_budget)
        self.eps = float(eps)
        reduction = int(reduction)
        residual_groups = int(residual_groups)

        if self.c_deep <= 0 or self.c_lateral <= 0 or self.scale <= 1:
            raise ValueError("OrthogonalComplementaryAlignUp channel counts and scale are invalid.")
        if reduction < 1 or residual_groups < 1:
            raise ValueError("reduction and residual_groups must be positive.")
        if self.max_gain < 0.0 or not 0.0 < self.max_residual_ratio <= 1.0:
            raise ValueError("max_gain must be non-negative and max_residual_ratio in (0, 1].")
        if not 0.0 <= self.candidate_floor <= 1.0 or self.eps <= 0.0:
            raise ValueError("candidate_floor must be in [0, 1] and eps positive.")

        self.hidden = max(16, min(64, min(self.c_deep, self.c_lateral) // reduction))
        self.residual_groups = max(1, math.gcd(math.gcd(self.hidden, self.c_deep), residual_groups))
        blur = torch.tensor(
            ((1.0, 2.0, 1.0), (2.0, 4.0, 2.0), (1.0, 2.0, 1.0)),
            dtype=torch.float32,
        )
        self.register_buffer("blur_kernel", (blur / blur.sum())[None, None], persistent=False)

        # Replacing a parameter-free nn.Upsample must not advance the downstream RNG stream.
        with torch.random.fork_rng(devices=[], enabled=True):
            self.deep_proj = nn.Conv2d(self.c_deep, self.hidden, 1, bias=False)
            self.lateral_proj = nn.Conv2d(self.c_lateral, self.hidden, 1, bias=False)
            self.candidate_gate = nn.Sequential(
                nn.Conv2d(3 * self.hidden + 4, self.hidden, 3, 1, 1, bias=True),
                nn.SiLU(inplace=True),
                nn.Conv2d(self.hidden, 1, 1, bias=True),
                nn.Sigmoid(),
            )
            self.residual_out = nn.Conv2d(
                self.hidden,
                self.c_deep,
                1,
                groups=self.residual_groups,
                bias=True,
            )
            nn.init.normal_(self.candidate_gate[-2].weight, mean=0.0, std=1e-3)
            nn.init.zeros_(self.candidate_gate[-2].bias)
            nn.init.kaiming_normal_(self.residual_out.weight, mode="fan_out", nonlinearity="linear")
            nn.init.zeros_(self.residual_out.bias)

        self.alpha_raw = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        self.record_diagnostics = False
        self.latest_diagnostics = {}

    def set_diagnostics(self, enabled=True):
        self.record_diagnostics = bool(enabled)
        if not self.record_diagnostics:
            self.latest_diagnostics = {}
        return self

    def _validate_inputs(self, deep, lateral):
        if deep.ndim != 4 or lateral.ndim != 4 or deep.shape[0] != lateral.shape[0]:
            raise ValueError("OrthogonalComplementaryAlignUp expects compatible NCHW tensors.")
        if deep.shape[1] != self.c_deep or lateral.shape[1] != self.c_lateral:
            raise ValueError("OrthogonalComplementaryAlignUp received unexpected channel counts.")
        if deep.device != lateral.device or deep.dtype != lateral.dtype:
            raise ValueError("OrthogonalComplementaryAlignUp inputs must share device and dtype.")
        expected = (deep.shape[-2] * self.scale, deep.shape[-1] * self.scale)
        if self.strict_scale and tuple(lateral.shape[-2:]) != expected:
            raise ValueError(f"Expected lateral size {expected}, got {tuple(lateral.shape[-2:])}.")

    def _blur(self, x):
        kernel = self.blur_kernel.to(device=x.device, dtype=x.dtype).repeat(self.hidden, 1, 1, 1)
        return F.conv2d(F.pad(x, (1, 1, 1, 1), mode="replicate"), kernel, groups=self.hidden)

    @staticmethod
    def _local_mean(x):
        return F.avg_pool2d(F.pad(x, (1, 1, 1, 1), mode="replicate"), 3, 1)

    @staticmethod
    def _zscore(x, eps):
        value = x.float()
        mean = value.mean(dim=(2, 3), keepdim=True)
        std = value.var(dim=(2, 3), keepdim=True, unbiased=False).add(eps).sqrt()
        return (value - mean) / std

    def _energy_budget(self, base, correction):
        if base.shape != correction.shape:
            raise ValueError("base and correction shapes differ.")
        # A sample-wise scalar budget preserves the per-pixel channel-space orthogonality.
        # Channel-wise scaling would generally destroy dot(base, correction) == 0.
        base_rms = base.float().square().mean(dim=(1, 2, 3), keepdim=True).sqrt()
        correction_rms = correction.float().square().mean(dim=(1, 2, 3), keepdim=True).sqrt()
        scale = torch.minimum(
            torch.ones_like(correction_rms),
            self.max_residual_ratio * base_rms / (correction_rms + self.eps),
        )
        if self.detach_budget:
            scale = scale.detach()
        return (correction.float() * scale).to(dtype=base.dtype), scale

    def _orthogonalize(self, base, correction):
        base_float = base.float()
        correction_float = correction.float()
        denominator = base_float.square().sum(dim=1, keepdim=True)
        projection = (correction_float * base_float).sum(dim=1, keepdim=True) / (denominator + self.eps)
        # For exactly zero base vectors the projection is zero, as required.
        return (correction_float - projection * base_float).to(dtype=correction.dtype)

    def compute_components(self, deep, lateral):
        self._validate_inputs(deep, lateral)
        target_size = lateral.shape[-2:]
        base = F.interpolate(deep, size=target_size, mode="nearest")
        deep_embed = F.interpolate(self.deep_proj(deep), size=target_size, mode="nearest")
        lateral_embed = self.lateral_proj(lateral)
        deep_low = self._blur(deep_embed)
        lateral_low = self._blur(lateral_embed)
        lateral_high = lateral_embed - lateral_low

        deep_norm = F.normalize(deep_low.float(), dim=1, eps=self.eps)
        lateral_norm = F.normalize(lateral_low.float(), dim=1, eps=self.eps)
        agreement = ((deep_norm * lateral_norm).sum(dim=1, keepdim=True) + 1.0).mul(0.5).clamp(0.0, 1.0)

        deep_energy = deep_low.float().abs().mean(dim=1, keepdim=True)
        lateral_energy = lateral_low.float().abs().mean(dim=1, keepdim=True)
        energy_scale = deep_energy + lateral_energy + self.eps
        missing_support = torch.sigmoid(((lateral_energy - deep_energy) / energy_scale).clamp(-8.0, 8.0))

        high_energy = lateral_high.float().abs().mean(dim=1, keepdim=True)
        relative_detail = torch.sigmoid(self._zscore(high_energy, self.eps))
        absolute_detail = 1.0 - torch.exp(
            -(high_energy / (deep_energy + lateral_energy + self.eps)).clamp(0.0, 8.0)
        )
        # A flat or exactly zero map must have zero detail support rather than a sigmoid default of 0.5.
        detail_support = (absolute_detail * relative_detail).clamp(0.0, 1.0)
        disagreement = (deep_embed.float() - lateral_embed.float()).abs().mean(dim=1, keepdim=True)
        discrepancy_support = torch.exp(-(disagreement / (deep_energy + lateral_energy + self.eps)).clamp(0.0, 8.0))

        analytic_candidate = (
            detail_support
            * (self.candidate_floor + (1.0 - self.candidate_floor) * missing_support)
            * (0.50 * agreement + 0.50 * discrepancy_support)
        ).clamp(0.0, 1.0)
        cues = torch.cat(
            (
                deep_embed,
                lateral_embed,
                (deep_embed - lateral_embed).abs(),
                agreement.to(deep.dtype),
                missing_support.to(deep.dtype),
                detail_support.to(deep.dtype),
                analytic_candidate.to(deep.dtype),
            ),
            dim=1,
        )
        candidate_gate = analytic_candidate.detach().to(deep.dtype) * self.candidate_gate(cues)

        raw_correction = candidate_gate * self.residual_out(lateral_high)
        zero_mean_correction = raw_correction - self._local_mean(raw_correction.float()).to(raw_correction.dtype)
        orthogonal_correction = self._orthogonalize(base, zero_mean_correction)
        correction, budget_scale = self._energy_budget(base, orthogonal_correction)
        return base, correction, candidate_gate, agreement, missing_support, detail_support, budget_scale

    def forward(self, x):
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise TypeError("OrthogonalComplementaryAlignUp expects [P4_deep, P3_lateral].")
        base, correction, candidate_gate, agreement, missing_support, detail_support, budget_scale = self.compute_components(
            x[0], x[1]
        )
        gain = (self.max_gain * torch.tanh(self.alpha_raw)).to(dtype=base.dtype)
        output = base + gain * correction
        if self.record_diagnostics:
            dot = (base.float() * correction.float()).sum(dim=1, keepdim=True)
            self.latest_diagnostics = {
                "gain": gain.detach(),
                "candidate_gate": candidate_gate.detach(),
                "agreement": agreement.detach(),
                "missing_support": missing_support.detach(),
                "detail_support": detail_support.detach(),
                "budget_scale": budget_scale.detach(),
                "orthogonal_dot": dot.detach(),
            }
        return output

class MissingAwareCandidateReactivator(nn.Module):
    """Recover P3 candidates supported by P2 detail and P4 context but weak in P3.

    Both spatial and channel releases are explicitly budgeted. A per-channel RMS budget
    and zero-start scalar preserve the high-precision base path while allowing bounded
    recovery of missed small/medium targets.
    """

    def __init__(
        self,
        c_shallow,
        c_context,
        channels,
        max_gain=0.08,
        spatial_rho=0.15,
        channel_rho=0.25,
        max_residual_ratio=0.12,
        unique_weight=0.30,
        reduction=4,
        detach_budget=True,
        eps=1e-6,
    ):
        super().__init__()
        self.c_shallow = int(c_shallow)
        self.c_context = int(c_context)
        self.channels = int(channels)
        self.max_gain = float(max_gain)
        self.spatial_rho = float(spatial_rho)
        self.channel_rho = float(channel_rho)
        self.max_residual_ratio = float(max_residual_ratio)
        self.unique_weight = float(unique_weight)
        self.detach_budget = bool(detach_budget)
        self.eps = float(eps)
        reduction = int(reduction)

        if min(self.c_shallow, self.c_context, self.channels) <= 0:
            raise ValueError("MissingAwareCandidateReactivator channel counts must be positive.")
        if self.max_gain < 0.0 or reduction < 1 or self.eps <= 0.0:
            raise ValueError("max_gain must be non-negative; reduction and eps positive.")
        for name, value in (
            ("spatial_rho", self.spatial_rho),
            ("channel_rho", self.channel_rho),
            ("max_residual_ratio", self.max_residual_ratio),
            ("unique_weight", self.unique_weight),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1].")
        if self.max_residual_ratio == 0.0:
            raise ValueError("max_residual_ratio must be positive.")

        hidden = max(self.channels // reduction, 16)
        blur = torch.tensor(
            ((1.0, 2.0, 1.0), (2.0, 4.0, 2.0), (1.0, 2.0, 1.0)),
            dtype=torch.float32,
        )
        self.register_buffer("blur_kernel", (blur / blur.sum())[None, None], persistent=False)

        self.shallow_proj = Conv(self.c_shallow, self.channels, 1, 1, act=False)
        self.context_proj = Conv(self.c_context, self.channels, 1, 1, act=False)
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(4, 8, 3, 1, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(8, 1, 1, bias=True),
            nn.Sigmoid(),
        )
        self.channel_gate = nn.Sequential(
            nn.Conv2d(3 * self.channels, hidden, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, self.channels, 1, bias=True),
            nn.Sigmoid(),
        )
        self.route_gate = nn.Sequential(
            Conv(3 * self.channels, hidden, 1, 1),
            nn.Conv2d(hidden, self.channels, 1, bias=True),
            nn.Sigmoid(),
        )
        self.out_proj = nn.Conv2d(self.channels, self.channels, 1, bias=True)
        nn.init.normal_(self.spatial_gate[-2].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.spatial_gate[-2].bias)
        nn.init.normal_(self.channel_gate[-2].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.channel_gate[-2].bias)
        nn.init.kaiming_normal_(self.out_proj.weight, mode="fan_out", nonlinearity="linear")
        nn.init.zeros_(self.out_proj.bias)

        self.alpha_raw = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        self.record_diagnostics = False
        self.latest_diagnostics = {}

    def set_diagnostics(self, enabled=True):
        self.record_diagnostics = bool(enabled)
        if not self.record_diagnostics:
            self.latest_diagnostics = {}
        return self

    @staticmethod
    def _local_mean(x):
        return F.avg_pool2d(F.pad(x, (1, 1, 1, 1), mode="replicate"), 3, 1)

    @staticmethod
    def _zscore(x, eps):
        value = x.float()
        mean = value.mean(dim=(2, 3), keepdim=True)
        std = value.var(dim=(2, 3), keepdim=True, unbiased=False).add(eps).sqrt()
        return (value - mean) / std

    def _blur_downsample(self, x):
        kernel = self.blur_kernel.to(device=x.device, dtype=x.dtype).repeat(self.c_shallow, 1, 1, 1)
        return F.conv2d(
            F.pad(x, (1, 1, 1, 1), mode="replicate"),
            kernel,
            stride=2,
            groups=self.c_shallow,
        )

    @staticmethod
    def _align(x, size, mode="bilinear"):
        if x.shape[-2:] == size:
            return x
        if mode == "nearest":
            return F.interpolate(x, size=size, mode=mode)
        return F.interpolate(x, size=size, mode=mode, align_corners=False)

    def _spatial_budget(self, gate):
        mean = gate.float().mean(dim=(2, 3), keepdim=True)
        scale = torch.minimum(
            torch.ones_like(mean),
            torch.full_like(mean, self.spatial_rho) / mean.clamp_min(self.eps),
        )
        if self.detach_budget:
            scale = scale.detach()
        return (gate.float() * scale).clamp(0.0, 1.0).to(gate.dtype), scale

    def _channel_budget(self, gate):
        mean = gate.float().mean(dim=1, keepdim=True)
        scale = torch.minimum(
            torch.ones_like(mean),
            torch.full_like(mean, self.channel_rho) / mean.clamp_min(self.eps),
        )
        if self.detach_budget:
            scale = scale.detach()
        return (gate.float() * scale).clamp(0.0, 1.0).to(gate.dtype), scale

    def _energy_budget(self, base, correction):
        base_rms = base.float().square().mean(dim=(2, 3), keepdim=True).sqrt()
        correction_rms = correction.float().square().mean(dim=(2, 3), keepdim=True).sqrt()
        scale = torch.minimum(
            torch.ones_like(correction_rms),
            self.max_residual_ratio * base_rms / (correction_rms + self.eps),
        )
        if self.detach_budget:
            scale = scale.detach()
        return (correction.float() * scale).to(dtype=base.dtype), scale

    def compute_components(self, shallow, base, context):
        if shallow.ndim != 4 or base.ndim != 4 or context.ndim != 4:
            raise ValueError("MissingAwareCandidateReactivator expects NCHW tensors.")
        if shallow.shape[0] != base.shape[0] or context.shape[0] != base.shape[0]:
            raise ValueError("MissingAwareCandidateReactivator batch sizes differ.")
        if shallow.shape[1] != self.c_shallow or context.shape[1] != self.c_context or base.shape[1] != self.channels:
            raise ValueError("MissingAwareCandidateReactivator received unexpected channel counts.")
        if shallow.device != base.device or context.device != base.device:
            raise ValueError("MissingAwareCandidateReactivator inputs must share device.")
        if shallow.dtype != base.dtype or context.dtype != base.dtype:
            raise ValueError("MissingAwareCandidateReactivator inputs must share dtype.")

        target_size = base.shape[-2:]
        shallow_low_raw = self._align(self._blur_downsample(shallow), target_size)
        shallow_avg_raw = F.adaptive_avg_pool2d(shallow, target_size)
        shallow_detail_raw = (F.adaptive_max_pool2d(shallow, target_size) - shallow_avg_raw).clamp_min(0.0)
        shallow_feature = self.shallow_proj(shallow_detail_raw)
        context_feature = self._align(self.context_proj(context), target_size, mode="nearest")

        shallow_detail_energy = shallow_detail_raw.float().abs().mean(dim=1, keepdim=True)
        shallow_low_energy = shallow_low_raw.float().abs().mean(dim=1, keepdim=True)
        shallow_ratio = (shallow_detail_energy / (shallow_low_energy + self.eps)).clamp(0.0, 8.0)
        shallow_support = ((1.0 - torch.exp(-shallow_ratio)) * torch.sigmoid(self._zscore(shallow_detail_energy, self.eps))).clamp(0.0, 1.0)

        context_energy = context_feature.float().abs().mean(dim=1, keepdim=True)
        context_reference = context_energy.mean(dim=(2, 3), keepdim=True)
        context_absolute = 1.0 - torch.exp(
            -(context_energy / (context_reference + self.eps)).clamp(0.0, 8.0)
        )
        context_support = (
            context_absolute * torch.sigmoid(self._zscore(context_energy, self.eps))
        ).clamp(0.0, 1.0)

        base_float = base.float()
        base_low = self._local_mean(base_float)
        base_detail = (base_float - base_low).abs().mean(dim=1, keepdim=True)
        base_energy = base_low.abs().mean(dim=1, keepdim=True) + base_detail
        base_reference = base_energy.mean(dim=(2, 3), keepdim=True)
        base_absolute = 1.0 - torch.exp(
            -(base_energy / (base_reference + self.eps)).clamp(0.0, 8.0)
        )
        base_support = (
            base_absolute * torch.sigmoid(self._zscore(base_energy, self.eps))
        ).clamp(0.0, 1.0)

        semantic_factor = context_support + self.unique_weight * (1.0 - context_support)
        missing_prior = (shallow_support * (1.0 - base_support) * semantic_factor).clamp(0.0, 1.0)
        spatial_cues = torch.cat((shallow_support, context_support, base_support, missing_prior), dim=1).detach().to(base.dtype)
        spatial_release, spatial_scale = self._spatial_budget(
            missing_prior.detach().to(base.dtype) * self.spatial_gate(spatial_cues)
        )

        def normalize_token(token):
            return token / (token.mean(dim=1, keepdim=True) + self.eps)

        shallow_token = shallow_feature.float().abs().mean(dim=(2, 3), keepdim=True)
        context_token = context_feature.float().abs().mean(dim=(2, 3), keepdim=True)
        base_token = base_float.abs().mean(dim=(2, 3), keepdim=True)
        channel_input = torch.cat(
            (normalize_token(shallow_token), normalize_token(context_token), normalize_token(base_token)), dim=1
        ).detach().to(base.dtype)
        channel_release, channel_scale = self._channel_budget(self.channel_gate(channel_input))

        route = self.route_gate(torch.cat((shallow_feature, context_feature, base), dim=1))
        raw_residual = self.out_proj(route * (shallow_feature + context_support.to(base.dtype) * context_feature))
        gated_residual = spatial_release * channel_release * raw_residual
        residual, energy_scale = self._energy_budget(base, gated_residual)
        return residual, spatial_release, channel_release, spatial_scale, channel_scale, energy_scale

    def forward(self, shallow, base, context):
        residual, spatial_release, channel_release, spatial_scale, channel_scale, energy_scale = self.compute_components(
            shallow, base, context
        )
        gain = (self.max_gain * torch.tanh(self.alpha_raw)).to(dtype=base.dtype)
        output = base + gain * residual
        if self.record_diagnostics:
            self.latest_diagnostics = {
                "gain": gain.detach(),
                "spatial_release": spatial_release.detach(),
                "channel_release": channel_release.detach(),
                "spatial_scale": spatial_scale.detach(),
                "channel_scale": channel_scale.detach(),
                "energy_scale": energy_scale.detach(),
            }
        return output


class MACRDSC3k2(DSC3k2):
    """Original P3 neck DSC3k2 plus bounded missing-candidate reactivation."""

    def __init__(
        self,
        c_shallow,
        c_fused,
        c_context,
        c2,
        n=1,
        dsc3k=False,
        e=0.5,
        max_gain=0.08,
        spatial_rho=0.15,
        channel_rho=0.25,
        max_residual_ratio=0.12,
        unique_weight=0.30,
        reduction=4,
        detach_budget=True,
        g=1,
        shortcut=True,
        k1=3,
        k2=7,
        d2=1,
    ):
        super().__init__(
            c1=c_fused,
            c2=c2,
            n=n,
            dsc3k=dsc3k,
            e=e,
            g=g,
            shortcut=shortcut,
            k1=k1,
            k2=k2,
            d2=d2,
        )
        self.c_shallow = int(c_shallow)
        self.c_fused = int(c_fused)
        self.c_context = int(c_context)
        self.c2 = int(c2)
        self.dsc3k_enabled = bool(dsc3k)
        with torch.random.fork_rng(devices=[], enabled=True):
            self.reactivator = MissingAwareCandidateReactivator(
                c_shallow=self.c_shallow,
                c_context=self.c_context,
                channels=self.c2,
                max_gain=max_gain,
                spatial_rho=spatial_rho,
                channel_rho=channel_rho,
                max_residual_ratio=max_residual_ratio,
                unique_weight=unique_weight,
                reduction=reduction,
                detach_budget=detach_budget,
            )

    def forward(self, x):
        if not isinstance(x, (list, tuple)) or len(x) != 3:
            raise ValueError("MACRDSC3k2 expects [P2_shallow, P3_fused, P4_context].")
        shallow, fused, context = x
        if shallow.ndim != 4 or fused.ndim != 4 or context.ndim != 4:
            raise ValueError("MACRDSC3k2 inputs must be NCHW tensors.")
        if shallow.shape[0] != fused.shape[0] or context.shape[0] != fused.shape[0]:
            raise ValueError("MACRDSC3k2 batch sizes differ.")
        if shallow.shape[1] != self.c_shallow or fused.shape[1] != self.c_fused or context.shape[1] != self.c_context:
            raise ValueError("MACRDSC3k2 received unexpected channel counts.")
        base = super().forward(fused)
        return self.reactivator(shallow, base, context)
