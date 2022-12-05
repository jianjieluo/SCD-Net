import argparse
import numpy as np
from tqdm import tqdm
import os

import basic_utils as utils
from scorer.base_scorer import BaseScorer, CACHE_DIR

def load_vocab(path):
    if len(path) == 0:
        return None
    vocab = ['.']
    with open(path, 'r') as fid:
        for line in fid:
            vocab.append(line.strip())
    return vocab


def decode_sequence(vocab, seq):
    N, T = seq.shape
    sents = []
    for n in range(N):
        words = []
        for t in range(T):
            ix = seq[n, t]
            if ix == 0:
                break
            words.append(vocab[ix])
        sent = ' '.join(words)
        sents.append(sent)
    return sents

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--last_kd', type=str, default='pred/ar_clip_rl_ensemble_kd_sents_ep_0.pkl')
    parser.add_argument('--new_pred', type=str, default='pred/diff_clip_rl_ep_36_ts_20_td_0.pkl')
    parser.add_argument('--out', type=str, default='output/update_ar_clip_rl_ensemble_kd_sents_ep_0_with_diff_clip_rl_ep_36_ts_20_td_0.pkl')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    output_dir = os.path.dirname(args.out)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    vocab = load_vocab(os.path.join(CACHE_DIR, 'vocabulary.txt'))
    scorer = BaseScorer()

    diff_pred = utils.load_pickle(args.new_pred)
    arrl_pred = utils.load_pickle(args.last_kd)

    total_count = len(arrl_pred)
    diff_count = 0
    diff_better_count = 0
    diff_better_list = []

    for imgid, a_token_ids in tqdm(arrl_pred.items()):
        d_token_ids = diff_pred[imgid]
        
        if (a_token_ids == d_token_ids).all():
            pass
        else:
            diff_count += 1
            # compare cider
            scorer_input = {
                "IDS": np.array([imgid] * 2),
                "G_SENTS_IDS": np.concatenate([a_token_ids, d_token_ids], axis=0),
            }
            rewards = scorer(scorer_input)['REWARDS']
            if rewards[0] < rewards[1]:
                diff_better_count += 1
                diff_better_list.append(
                    (imgid, d_token_ids)
                )

    print('Diff ratio: {} / {} = {}'.format(diff_count, total_count, round(diff_count/total_count, 2)))
    print('Diff better ratio: {} / {} = {}'.format(diff_better_count, total_count, round(diff_better_count/total_count, 2)))

    print("################# Update KD files #################")
    m_pred = utils.load_pickle(args.last_kd)
    for (imgid, d_tokens_ids) in tqdm(diff_better_list):
        m_pred[imgid] = d_tokens_ids
    print(len(m_pred))
    utils.save_pickle(m_pred, args.out)
    