# Copyright 2022 JD.com, Inc., JD AI
"""
@author: Jianjie Luo
@contact: jianjieluo.sysu@gmail.com
"""
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

from random import random

from xmodaler.config import configurable
from xmodaler.config import kfg
from xmodaler.functional import beta_linear_log_snr, alpha_cosine_log_snr, log_snr_to_alpha_sigma, right_pad_dims_to
from xmodaler.functional import pad_tensor, dict_to_cuda, flat_list_of_lists
from .transformer_enc_dec import TransformerEncoderDecoder
from .build import META_ARCH_REGISTRY

__all__ = ["BitDiffusionTransformerEncoderDecoder"]

@META_ARCH_REGISTRY.register()
class BitDiffusionTransformerEncoderDecoder(TransformerEncoderDecoder):
    @configurable
    def __init__(
        self,
        *,
        vocab_size,
        max_seq_len,
        token_embed,
        visual_embed,
        encoder,
        decoder,
        predictor,
        greedy_decoder,
        beam_searcher,
        v_predictor,

        log_snr
    ):
        super().__init__(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            token_embed=token_embed,
            visual_embed=visual_embed,
            encoder=encoder,
            decoder=decoder,
            predictor=predictor,
            greedy_decoder=greedy_decoder,
            beam_searcher=beam_searcher,
            v_predictor=v_predictor
        )
        self.log_snr = log_snr

    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)

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

    def _forward(self, batched_inputs):
        inputs = batched_inputs
        masks = self.get_extended_attention_mask(batched_inputs)
        inputs.update(masks)

        ve_out = self.visual_embed(batched_inputs)
        inputs.update(ve_out)

        if self.encoder is not None:
            encoder_out_v = self.encoder(inputs, mode='v')
            inputs.update(encoder_out_v)

        if self.decoder is not None:
            inputs = self.decoder.preprocess(inputs)

        # convert txt to bit representation
        input_ids_bit = self.token_embed.get_bit_repr(inputs[kfg.U_TOKENS_IDS])
        inputs.update({
            kfg.U_TOKENS_IDS_BIT: input_ids_bit,
            kfg.U_TARGET_IDS: input_ids_bit
        })

        # noise sample
        corrupt_out = self.noise_sample(inputs)
        inputs.update(corrupt_out)
        # noised_input_ids_bit = corrupt_out[kfg.U_TOKENS_IDS]

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly
        self_cond = None
        if random() < 0.5:
            with torch.no_grad():
                inputs = self._diff_decoder_forward(inputs)
                self_cond = inputs[kfg.U_LOGITS].detach_()
                del inputs[kfg.U_LOGITS]
        inputs.update({ kfg.SELF_COND: self_cond })

        inputs = self._diff_decoder_forward(inputs)
        return inputs

    def _diff_decoder_forward(self, inputs):
        te_out = self.token_embed(inputs) # inputs may have self_cond
        inputs.update(te_out)

        # predict and take gradient step
        decoder_out = self.decoder(inputs)
        inputs.update(decoder_out)

        # bert hidden_size -> bit dim
        tlogits = self.predictor(inputs)
        inputs.update(tlogits)

        return inputs

    def noise_sample(self, inputs):
        # corrupt bit repr, i.e. 14-dim vector
        batch_size = inputs[kfg.ATT_FEATS].shape[0]
        device = inputs[kfg.ATT_FEATS].device

        # sample random times
        times = torch.zeros((batch_size,), device = device).float().uniform_(0, 0.999)

        bit_token_embed = inputs[kfg.U_TOKENS_IDS_BIT]
        noise = torch.randn_like(bit_token_embed)

        noise_level = self.log_snr(times)
        padded_noise_level = right_pad_dims_to(bit_token_embed, noise_level)
        alpha, sigma =  log_snr_to_alpha_sigma(padded_noise_level) # 从 noise 那里去生成 alpha 和均值量

        noised_bit_token_embed = alpha * bit_token_embed + sigma * noise
        return {
            kfg.U_TOKENS_IDS_BIT: noised_bit_token_embed,
            kfg.TIME_STEP: noise_level
        }

    def preprocess_batch(self, batched_inputs):
        super_ret = super().preprocess_batch(batched_inputs)

        sample_per_sample = batched_inputs[0].get(kfg.SAMPLE_PER_SAMPLE, 1)
        ret = {}

        if kfg.C_TOKENS_IDS in batched_inputs[0]:
            c_tokens_ids = [x[kfg.C_TOKENS_IDS] for x in batched_inputs]
            if sample_per_sample > 1:
                c_tokens_ids = flat_list_of_lists(c_tokens_ids)
            c_tokens_ids = pad_tensor(c_tokens_ids, padding_value=0, use_mask=False)
            ret.update( { kfg.C_TOKENS_IDS: c_tokens_ids } )

        if kfg.RL_SAMPLE_FIX_TOKENS_IDS in batched_inputs[0]:
            rl_sample_fix_tokens_ids = [x[kfg.RL_SAMPLE_FIX_TOKENS_IDS] for x in batched_inputs]
            if sample_per_sample > 1:
                rl_sample_fix_tokens_ids = flat_list_of_lists(rl_sample_fix_tokens_ids)
            rl_sample_fix_tokens_ids = pad_tensor(rl_sample_fix_tokens_ids, padding_value=0, use_mask=False)
            ret.update( { kfg.RL_SAMPLE_FIX_TOKENS_IDS: rl_sample_fix_tokens_ids } )

        dict_to_cuda(ret)
        super_ret.update(ret)
        return super_ret