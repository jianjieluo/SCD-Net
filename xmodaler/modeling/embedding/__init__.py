# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Yehao Li
@contact: yehaoli.sysu@gmail.com
"""
from .build import build_embeddings
from .token_embed import TokenBaseEmbedding, RefTokenBaseEmbedding
from .visual_embed import VisualBaseEmbedding, VisualIdentityEmbedding
from .visual_embed_conv import TDConvEDVisualBaseEmbedding
from .visual_grid_embed import VisualGridEmbedding

from .bit_embed import BitEmbedding

__all__ = list(globals().keys())