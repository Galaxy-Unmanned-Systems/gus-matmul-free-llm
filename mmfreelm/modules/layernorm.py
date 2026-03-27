# -*- coding: utf-8 -*-

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


def activation_quant(x: torch.Tensor) -> torch.Tensor:
    scale = 127.0 / x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-5)
    return (x * scale).round().clamp(-128, 127) / scale


def weight_quant(w: torch.Tensor) -> torch.Tensor:
    scale = w.abs().mean().clamp_min(1e-5)
    quant = (w / scale).round().clamp(-1, 1) * scale
    return quant


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


class LayerNorm(nn.Module):

    def __init__(self, hidden_size: int, eps: float = 1e-6, elementwise_affine: bool = True) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.bias = nn.Parameter(torch.zeros(hidden_size))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None, prenorm: bool = False):
        weight = self.weight if self.weight is not None else torch.ones(x.shape[-1], device=x.device, dtype=x.dtype)
        return layer_norm_ref(x, weight, self.bias, residual=residual, eps=self.eps, prenorm=prenorm, upcast=True)


class RMSNorm(nn.Module):

    def __init__(self, hidden_size: int, eps: float = 1e-6, elementwise_affine: bool = True) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(hidden_size))
        else:
            self.register_parameter("weight", None)
        self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None, prenorm: bool = False):
        weight = self.weight if self.weight is not None else torch.ones(x.shape[-1], device=x.device, dtype=x.dtype)
        return rms_norm_ref(x, weight, None, residual=residual, eps=self.eps, prenorm=prenorm, upcast=True)


class LayerNormLinear(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        out_features: int,
        bias: bool = False,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.norm = LayerNorm(hidden_size, eps=eps)
        self.weight = nn.Parameter(torch.empty(out_features, hidden_size))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None) -> torch.Tensor:
        normed = self.norm(x, residual=residual)
        return F.linear(normed, self.weight, self.bias)


class RMSNormLinear(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        out_features: int,
        bias: bool = False,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.norm = RMSNorm(hidden_size, eps=eps)
        self.weight = nn.Parameter(torch.empty(out_features, hidden_size))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None) -> torch.Tensor:
        normed = self.norm(x, residual=residual)
        return F.linear(normed, self.weight, self.bias)
