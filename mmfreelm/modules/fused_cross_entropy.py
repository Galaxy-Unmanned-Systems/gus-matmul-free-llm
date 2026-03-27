# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton  # noqa: F401
    import triton.language as tl  # noqa: F401
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


class FusedCrossEntropyLoss(nn.Module):

    def __init__(
        self,
        ignore_index: int = -100,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
        logit_scale: float = 1.0,
        lse_square_scale: float = 0.0,
        inplace_backward: bool = False,
        process_group: Optional[object] = None,
    ) -> None:
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.logit_scale = logit_scale
        self.lse_square_scale = lse_square_scale
        self.inplace_backward = inplace_backward
        self.process_group = process_group

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        scaled_logits = logits * self.logit_scale
        loss = F.cross_entropy(
            scaled_logits,
            labels,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            label_smoothing=self.label_smoothing,
        )
        if self.lse_square_scale != 0.0:
            valid = labels != self.ignore_index
            if valid.any():
                lse = torch.logsumexp(scaled_logits[valid], dim=-1)
                z_loss = self.lse_square_scale * lse.square().mean()
                loss = loss + z_loss
        return loss
