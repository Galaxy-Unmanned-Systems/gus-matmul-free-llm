# -*- coding: utf-8 -*-

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton  # noqa: F401
    import triton.language as tl  # noqa: F401
    from torch.cuda.amp import custom_bwd, custom_fwd  # noqa: F401
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


def layer_norm_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    residual: torch.Tensor | None = None,
    eps: float = 1e-6,
    prenorm: bool = False,
    upcast: bool = False,
):
    dtype = x.dtype
    if upcast:
        x = x.float()
        weight = weight.float()
        bias = bias.float() if bias is not None else None
        residual = residual.float() if residual is not None else None
    if residual is not None:
        x = x + residual
    out = F.layer_norm(x.to(weight.dtype), x.shape[-1:], weight=weight, bias=bias, eps=eps).to(dtype)
    return out if not prenorm else (out, x.to(dtype))


def rms_norm_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    residual: torch.Tensor | None = None,
    eps: float = 1e-6,
    prenorm: bool = False,
    upcast: bool = False,
):
    dtype = x.dtype
    if upcast:
        x = x.float()
        weight = weight.float()
        bias = bias.float() if bias is not None else None
        residual = residual.float() if residual is not None else None
    if residual is not None:
        x = x + residual
    rstd = torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps)
    out = x * rstd * weight
    if bias is not None:
        out = out + bias
    out = out.to(dtype)
    return out if not prenorm else (out, x.to(dtype))


class FusedRMSNormSwishGate(nn.Module):

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(
        self,
        gate: torch.Tensor,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if residual is not None:
            x = x + residual
        x = x.float()
        rstd = torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + self.eps)
        normed = (x * rstd).to(gate.dtype) * self.weight.to(gate.dtype)
        gated = gate * torch.sigmoid(gate)
        return normed * gated
