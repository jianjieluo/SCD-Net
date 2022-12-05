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
from ..embedding import build_embeddings
from ..encoder import build_encoder, add_encoder_config
from .bit_diffusion_transformer_enc_dec import BitDiffusionTransformerEncoderDecoder
from .build import META_ARCH_REGISTRY

__all__ = ["BitDiffusionRefTransformerEncoderDecoder"]

@META_ARCH_REGISTRY.register()
class BitDiffusionRefTransformerEncoderDecoder(BitDiffusionTransformerEncoderDecoder):
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
        log_snr,

        r_token_embed,
        r_encoder
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
            v_predictor=v_predictor,
            log_snr=log_snr
        )
        self.r_token_embed = r_token_embed
        self.r_encoder = r_encoder

    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        
        tmp_cfg = cfg.clone()
        tmp_cfg.defrost()
        tmp_cfg.MODEL.TOKEN_EMBED.NAME = 'RefTokenBaseEmbedding'
        tmp_cfg.freeze()
        r_token_embed = build_embeddings(tmp_cfg, tmp_cfg.MODEL.TOKEN_EMBED.NAME)
        
        tmp_cfg = cfg.clone()
        tmp_cfg.defrost()
        tmp_cfg.MODEL.BERT.NUM_HIDDEN_LAYERS = 3
        tmp_cfg.freeze()
        r_encoder = build_encoder(tmp_cfg)

        ret.update({
            'r_token_embed': r_token_embed,
            'r_encoder': r_encoder
        })
        return ret

    def get_extended_attention_mask(self, batched_inputs):
        ret = super().get_extended_attention_mask(batched_inputs)

        rmasks = batched_inputs[kfg.R_TOKENS_MASKS]
        rmasks = rmasks.to(dtype=next(self.parameters()).dtype)
        rmasks = rmasks.unsqueeze(1).unsqueeze(2)
        ext_rmasks = (1.0 - rmasks) * -10000.0
        ret.update({
            kfg.EXT_R_TOKENS_MASKS: ext_rmasks
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

        # ref token embed
        if hasattr(self, "r_token_embed") and self.r_token_embed is not None:
            re_out = self.r_token_embed(inputs)
            inputs.update(re_out)

        # convert txt to bit representation
        input_ids_bit = self.token_embed.get_bit_repr(inputs[kfg.U_TOKENS_IDS])
        inputs.update({
            kfg.U_TOKENS_IDS_BIT: input_ids_bit,
            kfg.U_TARGET_IDS: input_ids_bit
        })

        if kfg.C_TOKENS_IDS in inputs:
            cascaded_ids_bit = self.token_embed.get_bit_repr(inputs[kfg.C_TOKENS_IDS])
            inputs.update({ kfg.C_TOKENS_IDS_BIT: cascaded_ids_bit })

        # noise sample
        corrupt_out = self.noise_sample(inputs)
        inputs.update(corrupt_out)

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

        # merge t and r, and forward r_encoder
        if hasattr(self, "r_encoder") and self.r_encoder is not None:
            seq_length = inputs[kfg.U_TOKENS_IDS].shape[-1]
            r_enc_inputs = {
                kfg.ATT_FEATS: torch.cat([inputs[kfg.U_TOKEN_EMBED], inputs[kfg.R_TOKEN_EMBED]], dim=1),
                kfg.EXT_ATT_MASKS: torch.cat([inputs[kfg.EXT_U_TOKENS_MASKS], inputs[kfg.EXT_R_TOKENS_MASKS]], dim=-1),
            }
            r_encoder_out = self.r_encoder(r_enc_inputs, mode='v')
            ref_awared_u_token_embed = r_encoder_out[kfg.ATT_FEATS][:, :seq_length, :].contiguous()
            inputs.update({
                kfg.U_TOKEN_EMBED: ref_awared_u_token_embed
            })

        # predict and take gradient step
        decoder_out = self.decoder(inputs)
        inputs.update(decoder_out)

        # bert hidden_size -> bit dim
        tlogits = self.predictor(inputs)
        inputs.update(tlogits)

        return inputs

    def preprocess_batch(self, batched_inputs):
        super_ret = super().preprocess_batch(batched_inputs)

        sample_per_sample = batched_inputs[0].get(kfg.SAMPLE_PER_SAMPLE, 1)
        ret = {}
        if kfg.R_TOKENS_IDS in batched_inputs[0]:
            r_tokens_ids = [x[kfg.R_TOKENS_IDS] for x in batched_inputs]
            if sample_per_sample > 1:
                r_tokens_ids = flat_list_of_lists(r_tokens_ids)
            r_tokens_ids = pad_tensor(r_tokens_ids, padding_value=0, use_mask=False)
            ret.update( { kfg.R_TOKENS_IDS: r_tokens_ids } )

        if kfg.R_TOKENS_MASKS in batched_inputs[0]:
            r_tokens_masks = [x[kfg.R_TOKENS_MASKS] for x in batched_inputs]
            if sample_per_sample > 1:
                r_tokens_masks = flat_list_of_lists(r_tokens_masks)
            r_tokens_masks = pad_tensor(r_tokens_masks, padding_value=0, use_mask=False)
            ret.update( { kfg.R_TOKENS_MASKS: r_tokens_masks } )

        if kfg.R_TOKENS_TYPE in batched_inputs[0]:
            r_tokens_type = [x[kfg.R_TOKENS_TYPE] for x in batched_inputs]
            if sample_per_sample > 1:
                r_tokens_type = flat_list_of_lists(r_tokens_type)
            r_tokens_type = pad_tensor(r_tokens_type, padding_value=0, use_mask=False)
            ret.update({ kfg.R_TOKENS_TYPE: r_tokens_type })

        dict_to_cuda(ret)
        super_ret.update(ret)
        return super_ret