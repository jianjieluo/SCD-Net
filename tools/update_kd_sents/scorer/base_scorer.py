import os
import sys
import numpy as np
import pickle
from .cider import Cider

__all__ = ['BaseScorer']

CACHE_DIR = '../../../open_source_dataset/mscoco_dataset'

class BaseScorer(object):
    def __init__(
        self,
    ): 
        self.types = ['Cider']
        self.scorers = [Cider(
            n=4,
            sigma=6.0,
            cider_cached=os.path.join(CACHE_DIR, "mscoco_train_cider.pkl")    
        )]
        self.eos_id = 0
        self.weights = [1.0]
        self.gts = pickle.load(open(os.path.join(CACHE_DIR, "mscoco_train_gts.pkl"), 'rb'), encoding='bytes')

    def get_sents(self, sent):
        words = []
        for word in sent:
            if word == self.eos_id:
                words.append(self.eos_id)
                break
            words.append(word)
        return words

    def __call__(self, batched_inputs):
        ids = batched_inputs["IDS"]
        res = batched_inputs["G_SENTS_IDS"].tolist()

        hypo = [self.get_sents(r) for r in res]
        gts = [self.gts[i] for i in ids]

        rewards_info = {}
        rewards = np.zeros(len(ids))
        for i, scorer in enumerate(self.scorers):
            score, scores = scorer.compute_score(gts, hypo)
            rewards += self.weights[i] * scores
            rewards_info[self.types[i]] = score
        rewards_info.update({ 'REWARDS': rewards })
        return rewards_info

