# Copyright 2022 JD.com, Inc., JD AI
"""
@author: Jianjie Luo
@contact: jianjieluo.sysu@gmail.com
"""
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable


from einops import rearrange, repeat
import numpy as np
from tqdm import tqdm
from functools import partial
from torch.special import expm1

from xmodaler.config import configurable
from xmodaler.config import kfg
from xmodaler.functional import bits_to_decimal, beta_linear_log_snr, alpha_cosine_log_snr, log_snr_to_alpha_sigma, right_pad_dims_to, log

from .decode_strategy import DecodeStrategy
from .build import DECODE_STRATEGY_REGISTRY

@DECODE_STRATEGY_REGISTRY.register()
class DiffusionSampler(DecodeStrategy):

    @configurable
    def __init__(
        self,
        timesteps,
        time_difference,
        log_snr,
        vocab_size,
        bit_scale,
        apply_sigmoid_to_pred,
        sample_noise,
        **base_kwargs,
    ):
        super().__init__(**base_kwargs)
        self.timesteps = timesteps
        self.time_difference = time_difference / timesteps
        self.log_snr = log_snr

        assert vocab_size == len(self.vocab)
        self.vocab_size = vocab_size
        self.bit_dim = int(np.ceil(np.log2(self.vocab_size)))
        self.bit_scale = bit_scale
        self.apply_sigmoid_to_pred = apply_sigmoid_to_pred
        self.sample_noise = sample_noise

    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        ret.update({
            "timesteps": cfg.DECODE_STRATEGY.DIFFUSION.TIMESTEPS,
            "time_difference": cfg.DECODE_STRATEGY.DIFFUSION.TIME_DIFFERENCE,
            "vocab_size": cfg.MODEL.VOCAB_SIZE,
            "bit_scale": cfg.MODEL.TOKEN_EMBED.BIT_SCALE,
            "apply_sigmoid_to_pred": 'SigmoidCrossEntropy' in cfg.LOSSES.NAMES,
            "sample_noise": cfg.DECODE_STRATEGY.DIFFUSION.SAMPLE_NOISE,
        })

        if cfg.DIFFUSION.NOISE_SCHEDULE == "linear":
            log_snr = beta_linear_log_snr
        elif cfg.DIFFUSION.NOISE_SCHEDULE == "cosine":
            log_snr = alpha_cosine_log_snr
        else:
            raise ValueError(f'invalid noise schedule {cfg.DIFFUSION.NOISE_SCHEDULE}')
        ret.update({
            "log_snr": log_snr
        })

        return ret

    def _forward(self, batched_inputs, model):
        batch_size = batched_inputs[kfg.ATT_FEATS].size(0)

        inputs = batched_inputs
        masks = model.get_extended_attention_mask(batched_inputs)
        inputs.update(masks)

        ve_out = model.visual_embed(batched_inputs)
        inputs.update(ve_out)

        encoder_out_v = model.encoder(inputs, mode='v')
        inputs.update(encoder_out_v)
        inputs = model.decoder.preprocess(inputs)

        if hasattr(model, "r_token_embed") and model.r_token_embed is not None:
            re_out = model.r_token_embed(inputs)
            inputs.update(re_out)

        if kfg.C_TOKENS_IDS in inputs:
            cascaded_ids_bit = model.token_embed.get_bit_repr(inputs[kfg.C_TOKENS_IDS])
            inputs.update({ kfg.C_TOKENS_IDS_BIT: cascaded_ids_bit })
        
        outputs, _ = self.ddpm_sample(model, (batch_size, self.max_seq_len, self.bit_dim), inputs)
        outputs = outputs.clamp(0., 10199.).long()

        return {
            kfg.IDS: batched_inputs[kfg.IDS],
            kfg.G_SENTS_IDS: outputs,
        }

    def get_sampling_timesteps(self, batch, *, device):
        times = torch.linspace(1., 0., self.timesteps + 1, device = device)
        times = repeat(times, 't -> b t', b = batch)
        times = torch.stack((times[:, :-1], times[:, 1:]), dim = 0)
        times = times.unbind(dim = -1)
        return times

    @torch.no_grad()
    def ddpm_sample(self, model, shape, inputs):
        (batch_size, seq_length, bit_dim) = shape
        device = inputs[kfg.ATT_FEATS].device

        time_pairs = self.get_sampling_timesteps(batch_size, device = device)

        txt = torch.randn(shape, device=device)

        inputs[kfg.U_TOKENS_IDS] = torch.zeros((batch_size, seq_length), device=device).long() # init_u_tokens_ids_for_pos_embed
        x_start = None
        for time, time_next in time_pairs:

            # add the time delay
            time_next = (time_next - self.time_difference).clamp(min = 0.)
            noise_cond = self.log_snr(time)

            # get predicted x0
            inputs.update({
                kfg.U_TOKENS_IDS_BIT: txt,
                kfg.TIME_STEP: noise_cond,
                kfg.SELF_COND: x_start,
            })
            x_start = model._diff_decoder_forward(inputs)[kfg.U_LOGITS]

            # clip x0
            if self.apply_sigmoid_to_pred:
                x_start = 2 * torch.sigmoid(x_start) - 1
            else:
                x_start.clamp_(-self.bit_scale, self.bit_scale)

            # get log(snr)
            log_snr = self.log_snr(time)  # gamma(t_now)
            log_snr_next = self.log_snr(time_next) # gamma(t_next)
            log_snr, log_snr_next = map(partial(right_pad_dims_to, txt), (log_snr, log_snr_next)) # reshape

            # get alpha sigma of time and next time
            alpha, sigma = log_snr_to_alpha_sigma(log_snr)
            alpha_next, sigma_next = log_snr_to_alpha_sigma(log_snr_next)

            # derive posterior mean and variance
            c = -expm1(log_snr - log_snr_next)   # Computes the exponential of the elements minus 1: y_{i} = e^{x_{i}} - 1

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

        return bits_to_decimal(txt, vocab_size=self.vocab_size, bits=self.bit_dim), None
