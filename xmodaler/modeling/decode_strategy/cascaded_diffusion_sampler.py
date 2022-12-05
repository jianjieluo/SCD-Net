# Copyright 2022 JD.com, Inc., JD AI
"""
@author: Jianjie Luo
@contact: jianjieluo.sysu@gmail.com
"""
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

import copy
from einops import rearrange, repeat
import numpy as np
from tqdm import tqdm
from functools import partial
from torch.special import expm1

from xmodaler.config import configurable
from xmodaler.config import kfg
from xmodaler.functional import bits_to_decimal, log_snr_to_alpha_sigma, right_pad_dims_to, log
from xmodaler.functional import decode_sequence, decode_sequence_bert
from .diffusion_sampler import DiffusionSampler
from .build import DECODE_STRATEGY_REGISTRY

@DECODE_STRATEGY_REGISTRY.register()
class CascadedDiffusionSampler(DiffusionSampler):

    @configurable
    def __init__(
        self,
        cas_timesteps,
        cas_time_difference,
        debug,
        **base_kwargs,
    ):
        super().__init__(**base_kwargs)
        self.cas_timesteps = cas_timesteps
        self.cas_time_difference = [td / ts for td,ts in zip(cas_time_difference, cas_timesteps)]
        self.debug = debug

    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        ret.update({
            "cas_timesteps": cfg.DECODE_STRATEGY.DIFFUSION.CAS_TIMESTEPS,
            "cas_time_difference": cfg.DECODE_STRATEGY.DIFFUSION.CAS_TIME_DIFFERENCE,
            "debug": cfg.DEBUG
        })
        return ret

    def forward(self, batched_inputs, output_sents, model, weights):
        ret = self._forward(batched_inputs, model, weights)
        if output_sents:
            if self.vocab:
                sents = decode_sequence(self.vocab, ret[kfg.G_SENTS_IDS])
            else:
                sep_token_id = self.bert_tokenizer.vocab["[SEP]"]
                sents = decode_sequence_bert(self.bert_tokenizer, ret[kfg.G_SENTS_IDS], sep_token_id)
            ret.update({ kfg.OUTPUT: sents })
        return ret


    def _forward(self, batched_inputs, model, weights):
        batch_size = batched_inputs[kfg.ATT_FEATS].size(0)

        inputs = batched_inputs
        masks = model.get_extended_attention_mask(batched_inputs)
        inputs.update(masks)

        inputs_arr = []
        # prepare all feats except diffusion
        for i in range(model.num):
            inputs_arr.append(copy.deepcopy(inputs))
            ve_out = model.models[i].visual_embed(batched_inputs)
            inputs_arr[i].update(ve_out)

            encoder_out_v = model.models[i].encoder(inputs_arr[i], mode='v')
            inputs_arr[i].update(encoder_out_v)
            inputs_arr[i] = model.models[i].decoder.preprocess(inputs_arr[i])

            if hasattr(model.models[i], "r_token_embed") and model.models[i].r_token_embed is not None:
                re_out = model.models[i].r_token_embed(inputs_arr[i])
                inputs_arr[i].update(re_out)

        # cascaded / ensemble diffusion
        for i in range(model.num):
            # upate td and ts
            self.timesteps = self.cas_timesteps[i]
            self.time_difference = self.cas_time_difference[i]

            if i > 0:
                # NOTE: stage1 has no cascaded sents
                cascaded_ids_bit = model.models[i].token_embed.get_bit_repr(outputs)
                inputs_arr[i].update({ kfg.C_TOKENS_IDS_BIT: cascaded_ids_bit })

            outputs, probs = self.ddpm_sample(model.models[:i+1], (batch_size, self.max_seq_len, self.bit_dim), inputs_arr[:i+1], weights[:i+1])
            outputs = outputs.clamp(0., 10199.).long()
            # mask-out all words after the first [EOS]
            mask = (torch.cumsum((outputs == self.eos_token_id), dim=-1) == 0).long()
            outputs = outputs * mask

            if self.debug:
                print("################ Model {} Diff Out ################".format(i))
                for sent in self.output_sents(outputs):
                    print(sent)

        if self.debug:
            print("################ Model Ensemble Final Diff Out ################")
            for sent in self.output_sents(outputs):
                print(sent)

        if self.debug:
            p_outputs = probs.argmax(-1)
            print("################ Model Ensemble Final Prob Out ################")
            for sent in self.output_sents(p_outputs):
                print(sent)

        return {
            kfg.IDS: batched_inputs[kfg.IDS],
            kfg.G_SENTS_IDS: outputs,
        }

    @torch.no_grad()
    def ddpm_sample(self, models, shape, inputs_arr, weights):
        (batch_size, seq_length, bit_dim) = shape
        device = inputs_arr[0][kfg.ATT_FEATS].device
        model_num = len(models)

        vocab_bit_buffer = models[0].predictor.vocab_bit_buffer
        def prob2bit(probs):
            return torch.matmul(probs, vocab_bit_buffer.expand(batch_size, -1, -1))

        time_pairs = self.get_sampling_timesteps(batch_size, device = device)
        txt = torch.randn(shape, device=device)

        # init kfg.U_TOKENS_IDS for pos embed
        for i in range(model_num):
            inputs_arr[i][kfg.U_TOKENS_IDS] = torch.zeros((batch_size, seq_length), device=device).long() # init_u_tokens_ids_for_pos_embed

        x_start = None
        x_start_probs = [None for _ in range(model_num)]
        weights_sum = np.sum(weights)
        for time, time_next in time_pairs:
            # add the time delay
            time_next = (time_next - self.time_difference).clamp(min = 0.)

            noise_cond = self.log_snr(time)

            for i in range(model_num):
                # get predicted x0
                inputs_arr[i].update({
                    kfg.U_TOKENS_IDS_BIT: txt,
                    kfg.TIME_STEP: noise_cond,
                    kfg.SELF_COND: x_start,
                })
                diff_output = models[i]._diff_decoder_forward(inputs_arr[i])
                x_start_probs[i] = F.softmax(diff_output[kfg.G_LOGITS], dim=-1) * weights[i]

            # fuse x_start_probs
            x_start_prob = torch.stack(x_start_probs).sum(0) / weights_sum
            x_start = prob2bit(x_start_prob)

            # clip x0
            if self.apply_sigmoid_to_pred:
                x_start = 2 * torch.sigmoid(x_start) - 1
            else:
                x_start.clamp_(-self.bit_scale, self.bit_scale)

            # get log(snr)
            log_snr = self.log_snr(time)
            log_snr_next = self.log_snr(time_next)
            log_snr, log_snr_next = map(partial(right_pad_dims_to, txt), (log_snr, log_snr_next))

            # get alpha sigma of time and next time
            alpha, sigma = log_snr_to_alpha_sigma(log_snr)
            alpha_next, sigma_next = log_snr_to_alpha_sigma(log_snr_next)

            # derive posterior mean and variance
            c = -expm1(log_snr - log_snr_next)
            mean = alpha_next * (txt * (1 - c) / alpha + c * x_start)

            if self.sample_noise:
                variance = (sigma_next ** 2) * c
                log_variance = log(variance)
                # get noise
                noise = torch.where(
                    rearrange(time_next > 0, 'b -> b 1 1'),
                    torch.randn_like(txt),
                    torch.zeros_like(txt)
                )
                txt = mean + (0.5 * log_variance).exp() * noise
            else:
                txt = mean

        return bits_to_decimal(txt, vocab_size=self.vocab_size, bits=self.bit_dim), x_start_prob
