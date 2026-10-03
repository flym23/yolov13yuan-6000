"""Process-local compatibility for the supplied YOLO12 qkv/7x7 checkpoint.

Attention equations follow Ultralytics v8.3.90 (AGPL-3.0):
https://github.com/ultralytics/ultralytics/blob/v8.3.90/ultralytics/nn/modules/block.py
Checkpoint module templates preserve its exact convolution/bias/normalization layout.
No existing project source files or other experiment processes are changed.
"""
from copy import deepcopy

import torch
from torch import nn


class CheckpointAAttn(nn.Module):
    templates = {}

    def __init__(self, dim, num_heads, area=1):
        super().__init__()
        key = (int(dim), int(num_heads), int(area))
        if key not in self.templates:
            raise RuntimeError(f'YOLO12 checkpoint has no attention template for {key}')
        template = self.templates[key]
        self.area, self.num_heads, self.head_dim = area, num_heads, dim // num_heads
        for name in ('qkv', 'proj', 'pe'):
            self.add_module(name, deepcopy(getattr(template, name)).requires_grad_(True))

    def forward(self, x):
        batch, channels, height, width = x.shape
        tokens = height * width
        qkv = self.qkv(x).flatten(2).transpose(1, 2)
        if self.area > 1:
            qkv = qkv.reshape(batch * self.area, tokens // self.area, channels * 3)
            batch, tokens, _ = qkv.shape
        query, key, value = (
            qkv.view(batch, tokens, self.num_heads, self.head_dim * 3)
            .permute(0, 2, 3, 1).split([self.head_dim] * 3, dim=2))
        attention = ((query.transpose(-2, -1) @ key) * (self.head_dim ** -0.5)).softmax(dim=-1)
        output = (value @ attention.transpose(-2, -1)).permute(0, 3, 1, 2)
        value = value.permute(0, 3, 1, 2)
        if self.area > 1:
            output = output.reshape(batch // self.area, tokens * self.area, channels)
            value = value.reshape(batch // self.area, tokens * self.area, channels)
            batch, tokens, _ = output.shape
        output = output.reshape(batch, height, width, channels).permute(0, 3, 1, 2).contiguous()
        value = value.reshape(batch, height, width, channels).permute(0, 3, 1, 2).contiguous()
        return self.proj(output + self.pe(value))


def install(weights):
    from ultralytics.nn.modules import block
    from ultralytics.nn.tasks import torch_safe_load
    checkpoint, _ = torch_safe_load(str(weights))
    model = (checkpoint.get('ema') or checkpoint['model']).float()
    templates = {}
    for module in model.modules():
        if module.__class__.__name__ not in ('AAttn', 'CheckpointAAttn'):
            continue
        if (not hasattr(module, 'qkv') or hasattr(module, 'qk')
                or module.pe.conv.kernel_size != (7, 7)):
            raise RuntimeError('Specified YOLO12 weight has an unexpected attention layout')
        key = (module.qkv.conv.in_channels, module.num_heads, module.area)
        module.__class__ = CheckpointAAttn
        templates.setdefault(key, deepcopy(module))
    if not templates:
        raise RuntimeError('Specified checkpoint has no original YOLO12 attention')
    CheckpointAAttn.templates = templates
    block.AAttn = CheckpointAAttn
    print('YOLO12_CHECKPOINT_COMPAT: qkv/7x7; process-local templates', sorted(templates), flush=True)


def verify_attention():
    """Check against PyTorch SDPA independently and exercise finite input gradients."""
    import torch.nn.functional as functional
    for (dim, heads, area), template in CheckpointAAttn.templates.items():
        module = deepcopy(template).eval()
        x = torch.randn(2, dim, 8, 8, requires_grad=True)
        batch, channels, height, width = x.shape
        qkv = module.qkv(x).flatten(2).transpose(1, 2)
        qkv = qkv.reshape(batch * area, height * width // area, heads, 3 * module.head_dim)
        q, k, v = qkv.permute(0, 2, 1, 3).chunk(3, dim=-1)
        attention = functional.scaled_dot_product_attention(q, k, v)
        output = attention.permute(0, 2, 1, 3).reshape(batch, height, width, channels).permute(0, 3, 1, 2)
        value = v.permute(0, 2, 1, 3).reshape(batch, height, width, channels).permute(0, 3, 1, 2)
        expected = module.proj(output + module.pe(value))
        actual = module(x)
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
        actual.square().mean().backward()
        if x.grad is None or not torch.isfinite(x.grad).all():
            raise RuntimeError('YOLO12 attention backward is nonfinite')
