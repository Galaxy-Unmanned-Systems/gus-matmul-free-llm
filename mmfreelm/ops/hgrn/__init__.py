# -*- coding: utf-8 -*-

from .naive import naive_recurrent_hgrn as chunk_hgrn
from .recurrent_fuse import fused_recurrent_hgrn

__all__ = [
    'chunk_hgrn',
    'fused_recurrent_hgrn'
]
