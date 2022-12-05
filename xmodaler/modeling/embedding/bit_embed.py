# Copyright 2022 JD.com, Inc., JD AI
"""
@author: Jianjie Luo
@contact: jianjieluo.sysu@gmail.com
"""
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
import math

from xmodaler.config import configurable
from xmodaler.config import kfg
from xmodaler.functional import decimal_to_bits
from .token_embed import TokenBaseEmbedding
from .build import EMBEDDING_REGISTRY

__all__ = ["BitEmbedding"]

# positional embeds
class LearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with learned sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

# helpers functions

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

@EMBEDDING_REGISTRY.register()
class BitEmbedding(TokenBaseEmbedding):
    @configurable
    def __init__(
        self, 
        *,
        dim: int,
        vocab_size: int, # include <BOS>/<EOS>

        bit_dim: int,
        bit_scale: float,
        time_embed,
        **kwargs
    ):
        kwargs.update({
            "dim": dim,
            "vocab_size": vocab_size
        })
        super(BitEmbedding, self).__init__(**kwargs)
        self.embeddings = kwargs.pop('embeddings', None)

        self.time_embed = time_embed
        self.bit_dim = bit_dim
        self.bit_scale = bit_scale
        self.vocab_size = vocab_size

    @classmethod
    def from_config(cls, cfg):
        kwargs = super().from_config(cfg)

        vocab_size = kwargs["vocab_size"]
        bit_dim = int(np.ceil(np.log2(vocab_size)))

        if len(cfg.DATALOADER.CASCADED_FILE) > 0:
            embeddings = nn.Linear(bit_dim*3, cfg.MODEL.TOKEN_EMBED.DIM, bias=False) # [self_cond, inputs_bit]
        else:
            embeddings = nn.Linear(bit_dim*2, cfg.MODEL.TOKEN_EMBED.DIM, bias=False) # [self_cond, inputs_bit]

        # time embeddings
        learned_sinusoidal_dim = 256
        time_dim = cfg.MODEL.BERT.HIDDEN_SIZE
        sinu_pos_emb = LearnedSinusoidalPosEmb(learned_sinusoidal_dim)
        fourier_dim = learned_sinusoidal_dim + 1

        time_embed = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, cfg.MODEL.BERT.HIDDEN_SIZE)
        )

        kwargs.update({
            "bit_dim": bit_dim,
            "embeddings": embeddings,
            "time_embed": time_embed,
            "bit_scale": cfg.MODEL.TOKEN_EMBED.BIT_SCALE
        })
        return kwargs

    def forward(self, batched_inputs):
        ret = {}
        assert kfg.U_TOKENS_IDS in batched_inputs

        u_tokens_ids = batched_inputs[kfg.U_TOKENS_IDS]
        u_tokens_ids_bit = batched_inputs[kfg.U_TOKENS_IDS_BIT]
        u_tokens_type = batched_inputs.get(kfg.U_TOKENS_TYPE, None)
        u_self_cond = batched_inputs.get(kfg.SELF_COND, None)
        time_info = batched_inputs[kfg.TIME_STEP]
        c_tokens_ids_bit = batched_inputs.get(kfg.C_TOKENS_IDS_BIT, None)
        
        u_token_embed = self._forward(u_tokens_ids, u_tokens_ids_bit, time_info, self_cond=u_self_cond, token_type_ids=u_tokens_type, c_tokens_ids_bit=c_tokens_ids_bit)
        ret.update({ kfg.U_TOKEN_EMBED: u_token_embed })

        return ret

    def get_bit_repr(self, input_ids):
        batch_size, seq_length = input_ids.shape
        input_ids = input_ids.view(batch_size, seq_length, 1, 1) # the same as img: batch_size x channel x height x weight
        input_ids_bit = decimal_to_bits(input_ids, vocab_size=self.vocab_size, bits=self.bit_dim) * self.bit_scale
        return input_ids_bit

    def _forward(self, input_ids, input_ids_bit, time_info, self_cond=None, token_type_ids=None, c_tokens_ids_bit=None):
        self_cond = default(self_cond, lambda: torch.zeros_like(input_ids_bit))
        input_ids_bit = torch.cat((self_cond, input_ids_bit), dim = -1)
        if c_tokens_ids_bit is not None:
            input_ids_bit = torch.cat((input_ids_bit, c_tokens_ids_bit), dim=-1)

        embeddings = self.embeddings(input_ids_bit)
        
        if self.embeddings_pos is not None:
            position_embeddings = self.embeddings_pos(input_ids)
            embeddings = embeddings + position_embeddings

        if self.time_embed is not None:
            time_embeddings = self.time_embed(time_info)
            embeddings = embeddings + time_embeddings.unsqueeze(1)

        if (self.embeddings_token_type is not None) and (token_type_ids is not None):
            embeddings_token_type = self.embeddings_token_type(token_type_ids)
            embeddings = embeddings + embeddings_token_type

        if self.embeddings_act is not None:
            embeddings = self.embeddings_act(embeddings)

        if self.embeddings_norm is not None:
            embeddings = self.embeddings_norm(embeddings)

        if self.embeddings_dropout is not None:
            embeddings = self.embeddings_dropout(embeddings)

        return embeddings