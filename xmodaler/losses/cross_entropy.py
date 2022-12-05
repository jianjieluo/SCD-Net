# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Yehao Li
@contact: yehaoli.sysu@gmail.com
"""
import torch
import torch.nn as nn
from xmodaler.config import configurable
from xmodaler.config import kfg
from .build import LOSSES_REGISTRY

@LOSSES_REGISTRY.register()
class CrossEntropy(nn.Module):
    @configurable
    def __init__(self, eos_id, loss_weight):
        super(CrossEntropy, self).__init__()
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)
        self.eos_id = eos_id
        self.loss_weight = loss_weight

    @classmethod
    def from_config(cls, cfg):
        return {
            'eos_id': cfg.SCORER.EOS_ID,
            'loss_weight': cfg.LOSSES.CLS_LOSS_WEIGHT
        }

    @classmethod
    def add_config(cls, cfg):
        pass

    def forward(self, outputs_dict):
        ret  = {}
        if kfg.G_LOGITS in outputs_dict:
            logits = outputs_dict[kfg.G_LOGITS]
            targets = outputs_dict[kfg.U_TOKENS_IDS]

            seq = outputs_dict[kfg.U_TOKENS_IDS]
            mask = (torch.cumsum((seq == self.eos_id), dim=-1) == 0)
            mask = torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1)

            logits = logits[mask, :]
            targets = targets[mask].long()
            loss = self.criterion(logits, targets)
            ret.update({ 'CrossEntropy loss(G)': loss * self.loss_weight })
            
        return ret
