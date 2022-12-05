"""
From original at https://github.com/aimagelab/meshed-memory-transformer/blob/master/models/beam_search/beam_search.py
Original copyright of AImageLab code below, modifications by Yehao Li, Copyright 2021.
"""
# Copyright (c) 2019, AImageLab
import copy
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from xmodaler.config import configurable
from xmodaler.config import kfg
from xmodaler.functional import expand_tensor, load_vocab, decode_sequence, decode_sequence_bert
from .decode_strategy import DecodeStrategy
from .build import DECODE_STRATEGY_REGISTRY

@DECODE_STRATEGY_REGISTRY.register()
class EnsembleBeamSearcher(DecodeStrategy):

    def _select(self, batch_size, beam_size, t, candidate_logprob):
        selected_logprob, selected_idx = torch.sort(candidate_logprob.view(batch_size, -1), -1, descending=True)
        selected_logprob, selected_idx = selected_logprob[:, :beam_size], selected_idx[:, :beam_size]
        return selected_idx, selected_logprob

    def _expand_state(self, states, selected_beam, batch_size, beam_size, cur_beam_size):
        for i in range(len(states)):
            shape = list(states[i].shape)
            beam = selected_beam
            for _ in shape[1:]:
                beam = beam.unsqueeze(-1)
            states[i] = torch.gather(states[i].view(*([batch_size, cur_beam_size] + shape[1:])), 1,
                beam.expand(*([batch_size, beam_size] + shape[1:])))
            states[i] = states[i].view(*([-1, ] + shape[1:]))

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
        out_size = batched_inputs.get('OUT_SIZE', 1)
        beam_size = self.beam_size
        log_probs = []
        selected_words = None
        seq_logprob = torch.zeros((batch_size, 1, 1)).cuda()
        seq_mask = torch.ones((batch_size, beam_size, 1)).cuda()
        wt = Variable(torch.zeros(batch_size, dtype=torch.long).cuda()) + self.bos_token_id

        inputs = batched_inputs
        masks = model.get_extended_attention_mask(batched_inputs)
        inputs.update(masks)

        inputs_arr = []
        for i in range(model.num):
            inputs_arr.append(copy.deepcopy(inputs))
            ve_out = model.models[i].visual_embed(batched_inputs)
            inputs_arr[i].update(ve_out)

            encoder_out_v = model.models[i].encoder(inputs_arr[i], mode='v')
            inputs_arr[i].update(encoder_out_v)
            inputs_arr[i] = model.models[i].decoder.preprocess(inputs_arr[i])
        
        outputs = []
        weights_sum = np.sum(weights)
        for t in range(self.max_seq_len):
            cur_beam_size = 1 if t == 0 else beam_size

            mprob = 0
            for i in range(model.num):
                inputs_arr[i].update({ kfg.G_TOKENS_IDS: wt, kfg.TIME_STEP: t })
                vt_out = model.models[i].token_embed(inputs_arr[i])
                inputs_arr[i].update(vt_out)

                encoder_out_t = model.models[i].encoder(inputs_arr[i], mode='t')
                inputs_arr[i].update(encoder_out_t)

                decoder_out = model.models[i].decoder(inputs_arr[i])
                inputs_arr[i].update(decoder_out)

                logit = model.models[i].predictor(inputs_arr[i])[kfg.G_LOGITS]
                prob = F.softmax(logit, dim=-1) * weights[i]
                mprob += prob

            mprob /= weights_sum
            word_logprob = mprob.log()
            word_logprob = word_logprob.view(batch_size, cur_beam_size, -1)
            candidate_logprob = seq_logprob + word_logprob
            
            # Mask sequence if it reaches EOS
            if t > 0:
                mask = (selected_words.view(batch_size, cur_beam_size) != self.eos_token_id).float().unsqueeze(-1)
                seq_mask = seq_mask * mask
                word_logprob = word_logprob * seq_mask.expand_as(word_logprob)
                old_seq_logprob = seq_logprob.expand_as(candidate_logprob).contiguous()
                old_seq_logprob[:, :, 1:] = -999
                candidate_logprob = seq_mask * candidate_logprob + old_seq_logprob * (1 - seq_mask)

            selected_idx, selected_logprob = self._select(batch_size, beam_size, t, candidate_logprob)
            selected_beam = torch.div(selected_idx, candidate_logprob.shape[-1], rounding_mode='floor')
            selected_words = selected_idx - selected_beam * candidate_logprob.shape[-1]
 
            if kfg.HISTORY_STATES in inputs_arr[0]:
                expand_keys = [kfg.HISTORY_STATES]
                if kfg.ENC_HISTORY_STATES in inputs_arr[0]:
                    expand_keys.append(kfg.ENC_HISTORY_STATES)
            else:
                expand_keys = [kfg.G_HIDDEN_STATES, kfg.G_CELL_STATES]

            for key in expand_keys:
                for i in range(model.num):
                    if key in inputs_arr[i]:
                        states = inputs_arr[i][key]
                        self._expand_state(states, selected_beam, batch_size, beam_size, cur_beam_size)
                        inputs_arr[i].update({ key: states })            

            seq_logprob = selected_logprob.unsqueeze(-1)
            seq_mask = torch.gather(seq_mask, 1, selected_beam.unsqueeze(-1))
            outputs = list(torch.gather(o, 1, selected_beam.unsqueeze(-1)) for o in outputs)
            outputs.append(selected_words.unsqueeze(-1))

            this_word_logprob = torch.gather(word_logprob, 1,
                selected_beam.unsqueeze(-1).expand(batch_size, beam_size, word_logprob.shape[-1]))
            this_word_logprob = torch.gather(this_word_logprob, 2, selected_words.unsqueeze(-1))
            log_probs = list(
                torch.gather(o, 1, selected_beam.unsqueeze(-1).expand(batch_size, beam_size, 1)) for o in log_probs)
            log_probs.append(this_word_logprob)
            selected_words = selected_words.view(-1, 1)
            wt = selected_words.squeeze(-1)

            if t == 0:
                expand_keys = { 
                    kfg.ATT_FEATS, 
                    kfg.GLOBAL_FEATS, 
                    kfg.ATT_MASKS, 
                    kfg.EXT_ATT_MASKS, 
                    kfg.P_ATT_FEATS, 
                    kfg.EXT_G_TOKENS_MASKS,
                    kfg.G_TOKENS_TYPE,
                    kfg.SEMANTICS_FEATS,
                    kfg.EXT_SEMANTICS_MASKS
                }
                for key in expand_keys:
                    for i in range(model.num):
                        if key in inputs_arr[i]:
                            if isinstance(inputs_arr[i][key], list):
                                inputs_arr[i][key] = inputs_arr[i][key][-1] # usually is ATT_FEATS in TDEN
                            tensor = expand_tensor(inputs_arr[i][key], beam_size)
                            inputs_arr[i].update({ key: tensor })

        seq_logprob, sort_idxs = torch.sort(seq_logprob, 1, descending=True)
        outputs = torch.cat(outputs, -1)
        outputs = torch.gather(outputs, 1, sort_idxs.expand(batch_size, beam_size, self.max_seq_len))
        log_probs = torch.cat(log_probs, -1)
        log_probs = torch.gather(log_probs, 1, sort_idxs.expand(batch_size, beam_size, self.max_seq_len))

        outputs = outputs.contiguous()[:, :out_size]
        log_probs = log_probs.contiguous()[:, :out_size]
        if out_size == 1:
            outputs = outputs.squeeze(1)
            log_probs = log_probs.squeeze(1)
        
        return {
            kfg.IDS: batched_inputs[kfg.IDS],
            kfg.G_SENTS_IDS: outputs,
            kfg.G_LOGP: log_probs
        }