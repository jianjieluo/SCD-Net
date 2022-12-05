# Copyright 2022 JD.com, Inc., JD AI
"""
@author: Jianjie Luo
@contact: jianjieluo.sysu@gmail.com
"""
import os
import copy
import pickle
import random
import torch
from tqdm import tqdm
import numpy as np
from .mscoco import MSCoCoDataset
from xmodaler.config import configurable
from xmodaler.config import kfg
from xmodaler.functional import read_np, dict_as_tensor
from ..build import DATASETS_REGISTRY

__all__ = ["MSCoCoDiffusionDataset"]

@DATASETS_REGISTRY.register()
class MSCoCoDiffusionDataset(MSCoCoDataset):
    @configurable
    def __init__(
        self,
        cas_rand_ratio,
        **kwargs
    ):
        super(MSCoCoDiffusionDataset, self).__init__(**kwargs)
        self.cas_rand_ratio = cas_rand_ratio
        
    @classmethod
    def from_config(cls, cfg, stage: str = "train"):
        ret = super().from_config(cfg, stage)
        ret['cas_rand_ratio'] = cfg.DATALOADER.CASCADED_SENT_RAND_RATIO
        return ret

    def load_data(self, cfg):
        datalist = super().load_data(cfg)

        if len(cfg.DATALOADER.CASCADED_FILE) > 0:
            cascaded_pred = pickle.load(open(cfg.DATALOADER.CASCADED_FILE, 'rb'), encoding='bytes')
            for i in range(len(datalist)):
                image_id = str(datalist[i]['image_id'])
                datalist[i]['cascaded_tokens_ids'] = cascaded_pred[image_id]

        return datalist

    def __call__(self, dataset_dict):
        ret = super().__call__(dataset_dict)
        ret[kfg.U_TOKENS_TYPE] = ret[kfg.G_TOKENS_TYPE]
        del ret[kfg.G_TOKENS_TYPE]
        dataset_dict = copy.deepcopy(dataset_dict)

        if 'cascaded_tokens_ids' in dataset_dict:
            cascaded_tokens_ids = dataset_dict['cascaded_tokens_ids']
            if cascaded_tokens_ids.shape[0] == 1:
                cascaded_tokens_ids = cascaded_tokens_ids.reshape(-1)

            if self.stage == 'train':
                ret[kfg.C_TOKENS_IDS] = [cascaded_tokens_ids for _ in range(self.seq_per_img)]
            else:
                ret[kfg.C_TOKENS_IDS] = [cascaded_tokens_ids]

        if self.stage != 'train':
            return ret

        u_tokens_ids = [torch.roll(item, -1, dims=0).long() for item in ret[kfg.G_TOKENS_IDS]]
        u_ret = {
            kfg.U_TOKENS_IDS: u_tokens_ids,
        }
        del ret[kfg.G_TOKENS_IDS]
        del ret[kfg.G_TARGET_IDS]
        ret.update(u_ret) # use U_TOKENS insted of G_TOKENS for non-autoregressive setting

        if kfg.C_TOKENS_IDS in ret and self.cas_rand_ratio > 0.0:
            # rand replace augmentation for cascaded_tokens_ids
            new_cascaded_tokens_ids_list = []
            for cascaded_tokens_ids in ret[kfg.C_TOKENS_IDS]:
                for i in range(len(cascaded_tokens_ids)):
                    if cascaded_tokens_ids[i] == 0:
                        break
                    
                    if random.random() < self.cas_rand_ratio:
                        # random replace the word in the i-th place
                        rand_word = random.randint(1, 10198) # Return a random integer N such that a <= N <= b
                        cascaded_tokens_ids[i] = int(rand_word)
                new_cascaded_tokens_ids_list.append(cascaded_tokens_ids)
            ret[kfg.C_TOKENS_IDS] = new_cascaded_tokens_ids_list
            
        dict_as_tensor(ret)
        return ret


@DATASETS_REGISTRY.register()
class MSCoCoDiffusionRefDataset(MSCoCoDiffusionDataset):
    @configurable
    def __init__(
        self,
        ref_sent_file,
        ref_rand_ratio,
        **kwargs,
    ):
        super(MSCoCoDiffusionRefDataset, self).__init__(**kwargs)
        self.ref_sent_file = ref_sent_file
        self.ref_rand_ratio = ref_rand_ratio
        
    @classmethod
    def from_config(cls, cfg, stage: str = "train"):
        ret = super().from_config(cfg, stage)
        ret['ref_sent_file'] = cfg.DATALOADER.REF_SENT_FILE
        ret['ref_rand_ratio'] = cfg.DATALOADER.REF_SENT_RAND_RATIO
        return ret

    def load_data(self, cfg):
        datalist = super().load_data(cfg)

        def _load_pkl_file(filepath):
            return pickle.load(open(filepath, 'rb'), encoding='bytes') if len(filepath) > 0 else None
        
        retrieval_sent_data = _load_pkl_file(self.ref_sent_file)
        for i in range(len(datalist)):
            image_id = int(datalist[i]['image_id'])
            if str(image_id) in retrieval_sent_data:
                datalist[i].update(retrieval_sent_data[str(image_id)])
            elif image_id in retrieval_sent_data:
                datalist[i].update(retrieval_sent_data[image_id])

        return datalist

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        ret = super().__call__(dataset_dict)

        if self.stage != 'train':
            # top-1
            r_tokens_ids = [ dataset_dict['r_tokens_ids'][0,:].astype(np.int64) ]
            r_tokens_mask = [ dataset_dict['r_tokens_mask'][0,:].astype(np.int64) ]
            r_tokens_type = [ np.zeros((len(dataset_dict['r_tokens_ids'][0,:]), ), dtype=np.int64) ]

        else:
            sent_num = len(dataset_dict['r_tokens_ids'])
            if sent_num >= self.seq_per_img:
                selects = random.sample(range(sent_num), self.seq_per_img)
            else:
                selects = random.choices(range(sent_num), k = (self.seq_per_img - sent_num))
                selects += list(range(sent_num))

            r_tokens_ids = [ dataset_dict['r_tokens_ids'][i,:].astype(np.int64) for i in selects ]
            r_tokens_mask = [ dataset_dict['r_tokens_mask'][i,:].astype(np.int64) for i in selects ]
            r_tokens_type = [ np.zeros((len(dataset_dict['r_tokens_ids'][i,:]), ), dtype=np.int64) for i in selects ]

            if self.ref_rand_ratio > 0.0:
                # rand replace augmentation for r_tokens_ids
                # ori_r_tokens_ids = copy.deepcopy(r_tokens_ids)
                for tokens, mask in zip(r_tokens_ids, r_tokens_mask):
                    seq_length = np.sum(mask)
                    for i in range(seq_length):
                        if i == 0:
                            # skip [SEP] 0 in the first place
                            continue
                        
                        if random.random() < self.ref_rand_ratio:
                            # random replace the word in the i-th place
                            rand_word = random.randint(1, 10198) # Return a random integer N such that a <= N <= b
                            tokens[i] = int(rand_word)

        ret.update({
            kfg.R_TOKENS_IDS: r_tokens_ids,
            kfg.R_TOKENS_MASKS: r_tokens_mask,
            kfg.R_TOKENS_TYPE: r_tokens_type
        })
        dict_as_tensor(ret)
        return ret


@DATASETS_REGISTRY.register()
class MSCoCoDiffusionKDDataset(MSCoCoDiffusionRefDataset):
    @configurable
    def __init__(
        self,
        kd_pred_file,
        use_kd_tokens_as_input,
        force_guided,
        **kwargs
    ):
        super(MSCoCoDiffusionKDDataset, self).__init__(**kwargs)
        self.kd_pred_file = kd_pred_file
        self.use_kd_tokens_as_input = use_kd_tokens_as_input
        self.force_guided = force_guided
        
    @classmethod
    def from_config(cls, cfg, stage: str = "train"):
        ret = super().from_config(cfg, stage)
        ret["kd_pred_file"] = cfg.DATALOADER.KD_PRED_FILE
        ret["use_kd_tokens_as_input"] = cfg.DATALOADER.USE_KD_TOKENS_AS_INPUT
        ret["force_guided"] = cfg.DATALOADER.FORCE_GUIDED
        return ret

    def load_data(self, cfg):
        datalist = super().load_data(cfg)

        def _load_pkl_file(filepath):
            return pickle.load(open(filepath, 'rb'), encoding='bytes') if len(filepath) > 0 else None
        
        kd_pred_data = _load_pkl_file(self.kd_pred_file)
        for i in range(len(datalist)):
            image_id = int(datalist[i]['image_id'])
            if str(image_id) in kd_pred_data:
                datalist[i].update({"kd_tokens_ids": kd_pred_data[str(image_id)].reshape(-1)})
            elif image_id in kd_pred_data:
                datalist[i].update({"kd_tokens_ids": kd_pred_data[image_id].reshape(-1)})

        return datalist

    def __call__(self, dataset_dict):
        ret = super().__call__(dataset_dict)

        if self.stage != 'train':
            return ret

        # load KD data
        dataset_dict = copy.deepcopy(dataset_dict)
        kd_tokens_ids = dataset_dict['kd_tokens_ids']
        mask = (torch.cumsum((torch.from_numpy(kd_tokens_ids) == 0), dim=-1) == 0)
        mask = torch.cat([mask.new_ones(1), mask[:-1]], dim=-1).long()

        kd_tokens_ids = [kd_tokens_ids for _ in range(self.seq_per_img)]
        if self.force_guided:
            ret[kfg.RL_SAMPLE_FIX_TOKENS_IDS] = kd_tokens_ids
        if self.use_kd_tokens_as_input:
            ret[kfg.U_TOKENS_IDS] = kd_tokens_ids

        dict_as_tensor(ret)
        return ret
