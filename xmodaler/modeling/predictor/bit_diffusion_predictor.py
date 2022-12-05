# Copyright 2022 JD.com, Inc., JD AI
"""
@author: Jianjie Luo
@contact: jianjieluo.sysu@gmail.com
"""
import torch
from torch import nn
import numpy as np

from .base_predictor import BasePredictor

from xmodaler.functional import decimal_to_bits

from xmodaler.config import configurable
from xmodaler.config import kfg
from .build import PREDICTOR_REGISTRY

__all__ = ["BitDiffusionPredictor"]

@PREDICTOR_REGISTRY.register()
class BitDiffusionPredictor(BasePredictor):
    @configurable
    def __init__(
        self,
        *,
        hidden_size: int,
        vocab_size: int,   # include <BOS>/<EOS>
        dropout: float,
        vocab_bit_buffer: torch.tensor
    ):
        super(BitDiffusionPredictor, self).__init__(
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            dropout=dropout
        )
        self.register_buffer('vocab_bit_buffer', vocab_bit_buffer, persistent=False)

    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)

        # generate buffer for UINT8-Rand
        bit_dim = int(np.ceil(np.log2(ret["vocab_size"])))
        bit_scale = cfg.MODEL.TOKEN_EMBED.BIT_SCALE
        vocab_size = ret["vocab_size"]

        vocab_inds = torch.arange(0, vocab_size).long().view(1, vocab_size, 1, 1)
        vocab_bit_buffer = decimal_to_bits(vocab_inds, vocab_size=vocab_size, bits=bit_dim) * bit_scale
        ret.update({
            "vocab_bit_buffer": vocab_bit_buffer
        })
        return ret

    @classmethod
    def add_config(cls, cfg):
        pass

    def forward(self, batched_inputs):
        hidden_states = batched_inputs[kfg.U_HIDDEN_STATES]
        if isinstance(hidden_states, list):
            hidden_states = hidden_states[-1]
        if self.dropout:  
            hidden_states = self.dropout(hidden_states)
        logits = self.logits(hidden_states)

        batch_size = logits.shape[0]
        buffer_probs = nn.Softmax(dim=-1)(logits)
        logits_w = torch.matmul(buffer_probs, self.vocab_bit_buffer.expand(batch_size, -1, -1))

        return { 
            kfg.U_LOGITS: logits_w,
            kfg.G_LOGITS: logits
        }