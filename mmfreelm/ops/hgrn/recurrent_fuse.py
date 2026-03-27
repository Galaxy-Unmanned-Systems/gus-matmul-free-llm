# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Tuple

import torch

try:
    import triton  # noqa: F401
    import triton.language as tl  # noqa: F401
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

from mmfreelm.ops.hgrn.naive import naive_recurrent_hgrn


def fused_recurrent_hgrn(
    x: torch.Tensor,
    g: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor | None]:
    if initial_state is not None:
        initial_state = initial_state.detach()
    return naive_recurrent_hgrn(
        x,
        g,
        initial_state=initial_state,
        output_final_state=output_final_state,
    )
