from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton  # noqa: F401
    import triton.language as tl  # noqa: F401
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

from mmfreelm.modules import RMSNorm


def activation_quant(x: torch.Tensor) -> torch.Tensor:
    scale = 127.0 / x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-5)
    return (x * scale).round().clamp(-128, 127) / scale


def weight_quant(w: torch.Tensor) -> torch.Tensor:
    scale = w.abs().mean().clamp_min(1e-5)
    return (w / scale).round().clamp(-1, 1) * scale


class BitLinearFallback(nn.Linear):

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__(in_features, out_features, bias=bias)
        self.norm = RMSNorm(in_features, eps=1e-8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm(x)
        x_quant = x_norm + (activation_quant(x_norm) - x_norm).detach()
        scale = self.weight.abs().mean().clamp_min(1e-5)
        w_round = (self.weight / scale).round().clamp(-1, 1) * scale
        w_q = self.weight + (w_round - self.weight).detach()
        return F.linear(x_quant, w_q, self.bias)


class BitLinear(BitLinearFallback):
    pass


class FusedBitLinear(BitLinearFallback):
    pass
