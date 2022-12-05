# Copyright 2022 JD.com, Inc., JD AI
"""
@author: Jianjie Luo
@contact: jianjieluo.sysu@gmail.com
"""
import copy
import tqdm
import weakref

import torch
from .bit_diffusion_trainer import BitDiffusionTrainer
from xmodaler.config import kfg
import xmodaler.utils.comm as comm
from xmodaler.checkpoint import XmodalerCheckpointer
from xmodaler.modeling.meta_arch.ensemble import Ensemble
from .build import ENGINE_REGISTRY

__all__ = ['BitDiffusionCascader']

@ENGINE_REGISTRY.register()
class BitDiffusionCascader(BitDiffusionTrainer):
    def __init__(self, cfg):
        super(BitDiffusionCascader, self).__init__(cfg)
        models = []
        ema_models = []
        num_models = len(cfg.MODEL.ENSEMBLE_WEIGHTS)
        assert num_models > 0, "cfg.MODEL.ENSEMBLE_WEIGHTS is empty"

        for i in range(num_models):
            if i == 0:
                # stage 1
                models.append(self.model)
                if cfg.MODEL.USE_EMA:
                    ema_models.append(copy.deepcopy(self.model))
            else:
                # stage 2, 3: rebuild bit_embed
                tmp_cfg = cfg.clone()
                tmp_cfg.defrost()
                tmp_cfg.DATALOADER.CASCADED_FILE = "trigger_bit_embed"
                tmp_cfg.freeze()
                model = self.build_model(tmp_cfg)
                models.append(model)
                if cfg.MODEL.USE_EMA:
                    ema_models.append(copy.deepcopy(model))

            # load model weights
            checkpointer = XmodalerCheckpointer(
                models[i],
                cfg.OUTPUT_DIR,
                trainer=weakref.proxy(self),
            )
            checkpointer.resume_or_load(cfg.MODEL.ENSEMBLE_WEIGHTS[i], resume=False)

            if cfg.MODEL.USE_EMA:
                # load ema_weights
                state_dict = torch.load(cfg.MODEL.ENSEMBLE_WEIGHTS[i], map_location='cpu')
                ema_weights = state_dict['trainer']['ema']
                if comm.get_world_size() == 1:
                    start_str = 'module.module.'
                    ema_weights = {k[len(start_str):] if k.startswith(start_str) else k : v for k,v in ema_weights.items()}
                ema_models[i].load_state_dict(ema_weights, strict=True)

        self.model = Ensemble(models, cfg)
        if cfg.MODEL.USE_EMA:
            self.ema = Ensemble(ema_models, cfg)
        else:
            self.ema = None

    def resume_or_load(self, resume=True):
        pass
