# Copyright 2022 JD.com, Inc., JD AI
"""
@author: Jianjie Luo
@contact: jianjieluo.sysu@gmail.com
"""
import tqdm
import os
import pickle

import time
import torch
from .defaults import DefaultTrainer
from xmodaler.config import kfg
from xmodaler.functional import bits_to_decimal
import xmodaler.utils.comm as comm
from .build import ENGINE_REGISTRY


__all__ = ['BitDiffusionTrainer']

@ENGINE_REGISTRY.register()
class BitDiffusionTrainer(DefaultTrainer):
    def __init__(self, cfg):
        super(BitDiffusionTrainer, self).__init__(cfg)
        self.debug = cfg.DEBUG
        

    def run_step(self):
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        """
        If you want to do something with the data, you can wrap the dataloader.
        """
        try:
            data = next(self._train_data_loader_iter)
        except StopIteration:
            if comm.get_world_size() > 1:
                self.train_data_loader.sampler.set_epoch(self.iter//self.iters_per_epoch)
            self._train_data_loader_iter = iter(self.train_data_loader)
            data = next(self._train_data_loader_iter)

        data_time = time.perf_counter() - start
        data = comm.unwrap_model(self.model).preprocess_batch(data)
        if self.debug:
            origin_targets = data[kfg.U_TOKENS_IDS]
            origin_targets_str = self.model.beam_searcher.output_sents(origin_targets.view(-1, 20))

        outputs_dict = self.model(data)

        if self.debug:
            image_ids = outputs_dict[kfg.IDS]
            targets_str = self.decode_bit_str(outputs_dict[kfg.U_TARGET_IDS])
            predict_str = self.decode_bit_str(outputs_dict[kfg.U_LOGITS])

            for image_id, o_target, target, pred in zip(image_ids, origin_targets_str, targets_str, predict_str):
                print("{}: \nPred:\t{}\nTarget\t{}\nGroundT\t{}\n".format(image_id, pred, target, o_target))

        losses_dict = {}
        for loss in self.losses:
            loss_dict = loss(outputs_dict)
            losses_dict.update(loss_dict)

        losses = [losses_dict[k] for k in losses_dict if 'acc' not in k]
        losses = sum(losses)

        """
        If you need to accumulate gradients or do something similar, you can
        wrap the optimizer with your custom `zero_grad()` method.
        """
        self.optimizer.zero_grad()
        losses.backward()

        self._write_metrics(losses_dict, data_time)

        self.optimizer.step()
        if self.ema is not None:
            self.ema.update(self.model)

    def decode_bit_str(self, input_ids_bit):
        target_ids = bits_to_decimal(input_ids_bit, vocab_size=10200, bits=14)
        target_ids = target_ids.clamp(0., 10199.).long()
        outputs_str = self.model.beam_searcher.output_sents(target_ids)
        return outputs_str

    @classmethod
    def test(cls, cfg, model, test_data_loader, evaluator, epoch):
        if cfg.DATALOADER.INFERENCE_TRAIN == False:
            return super().test(cfg, model, test_data_loader, evaluator, epoch)

        else:
            model.eval()
            results = {}
            with torch.no_grad():
                for data in tqdm.tqdm(test_data_loader):
                    data = comm.unwrap_model(model).preprocess_batch(data)
                    ids = data[kfg.IDS]

                    res = model(data, use_beam_search=True, output_sents=False)

                    g_sents_ids = res[kfg.G_SENTS_IDS]
                    # mask-out all words after the first [EOS]
                    eos_id = comm.unwrap_model(model).beam_searcher.eos_token_id
                    mask = (torch.cumsum((g_sents_ids == eos_id), dim=-1) == 0).long()
                    g_sents_ids = g_sents_ids * mask
                    g_sents_ids = g_sents_ids.cpu().numpy()

                    for id, g_sent_ids in zip(ids, g_sents_ids):
                        results[id] = g_sent_ids.reshape(1, -1)
            
            # save results in the output_dir
            if evaluator.output_dir is not None:
                filename = 'ep_{}_ts_{}_td_{}.pkl'.format(epoch, int(cfg.DECODE_STRATEGY.DIFFUSION.TIMESTEPS), int(cfg.DECODE_STRATEGY.DIFFUSION.TIME_DIFFERENCE))
                file_path = os.path.join(evaluator.output_dir, filename)
                with open(file_path, "wb") as f:
                    pickle.dump(results, f, protocol=4)

            model.train()
            return ''