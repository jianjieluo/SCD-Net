# Copyright 2022 JD.com, Inc., JD AI
"""
@author: Jianjie Luo
@contact: jianjieluo.sysu@gmail.com
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from xmodaler.config import configurable
from xmodaler.config import kfg
from .build import LOSSES_REGISTRY

@LOSSES_REGISTRY.register()
class MSELoss(nn.Module):
    @configurable
    def __init__(self, eos_id):
        super(MSELoss, self).__init__()
        self.eos_id = eos_id

    @classmethod
    def from_config(cls, cfg):
        return {
            'eos_id': cfg.SCORER.EOS_ID
        }

    @classmethod
    def add_config(cls, cfg):
        pass

    def forward(self, outputs_dict):
        ret  = {}
        if kfg.U_LOGITS in outputs_dict:
            logits = outputs_dict[kfg.U_LOGITS]
            targets = outputs_dict[kfg.U_TARGET_IDS] # already be encoded to bit representation in BitEmbed layer

            seq = outputs_dict[kfg.U_TOKENS_IDS]
            mask = (torch.cumsum((seq == self.eos_id), dim=-1) == 0)
            mask = torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1)

            logits = logits[mask, :]
            targets = targets[mask, :]
            loss = F.mse_loss(logits, targets)
            ret.update({'MSE loss(U)': loss})
            
        return ret
