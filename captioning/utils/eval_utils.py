from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import json
from json import encoder
import random
import string
import time
import os
import pdb
import sys
from . import misc as utils
sys.path.append("coco-caption")
from pycocoevalcap.eval_fic import FICScorer

bad_endings = ['a', 'an', 'the', 'in', 'for', 'at', 'of', 'with', 'before', 'after', 'on', 'upon', 'near', 'to', 'is',
               'are', 'am']
bad_endings += ['the']


def count_bad(sen):
    sen = sen.split(' ')
    if sen[-1] in bad_endings:
        return 1
    else:
        return 0


def language_eval(dataset, preds, preds_n, eval_kwargs, split):
    references = []  # references (true captions) for calculating BLEU-4 score
    hypotheses = []  # hypotheses (predictions)
    model_id = eval_kwargs['id']

    for j in range(len(preds)):
        references.append({'caption': preds[j]['gt']})
        hypotheses.append({'caption': preds[j]['caption']})

    assert len(references) == len(hypotheses)
    scorer = FICScorer()
    ids = [str(k) for k in range(len(hypotheses))]
    hypo = {}
    refe = {}
    for k in range(len(hypotheses)):
        hypo[str(k)] = [hypotheses[k]]
        refe[str(k)] = [references[k]]
    final_scores = scorer.score(refe, hypo, ids)

    cache_path = os.path.join('eval_results/', 'cache_' + model_id + '_' + split + '.json')
    json.dump(preds, open(cache_path, 'w'), indent = 4, ensure_ascii = False)
    # serialize to temporary json file. Sigh, COCO API..

    cache_path = os.path.join('eval_results/', 'cache_result_' + model_id + '_' + split + '.json')
    json.dump(final_scores, open(cache_path, 'w'))  # serialize to temporary json file. Sigh, COCO API..
    return final_scores


def eval_split(model, crit, loader, eval_kwargs={}):
    verbose = eval_kwargs.get('verbose', False)
    verbose_beam = eval_kwargs.get('verbose_beam', 0)
    verbose_loss = eval_kwargs.get('verbose_loss', 1)
    num_images = eval_kwargs.get('num_images', eval_kwargs.get('val_images_use', -1))
    split = eval_kwargs.get('split', 'val')
    lang_eval = eval_kwargs.get('language_eval', 0)
    dataset = eval_kwargs.get('dataset', 'coco')
    sample_n = eval_kwargs.get('sample_n', 1)
    remove_bad_endings = eval_kwargs.get('remove_bad_endings', 0)
    os.environ["REMOVE_BAD_ENDINGS"] = str(remove_bad_endings) # Use this nasty way to make other code clean since it's a global configuration
    device = eval_kwargs.get('device', 'cuda')

    # Make sure in the evaluation mode
    model.eval()
    loader.reset_iterator(split)

    n = 0
    loss = 0
    loss_sum = 0
    loss_evals = 1e-8
    predictions = []
    n_predictions = []  # when sample_n > 1
    while True:
        data = loader.get_batch(split)
        n = n + len(data['infos'])

        tmp = [data['img_feats'],  data['labels'], data['masks']]
        tmp = [_.to(device) if _ is not None else _ for _ in tmp]
        img_feats, labels, masks = tmp
        if labels is not None and verbose_loss:
            # forward the model to get loss
            with torch.no_grad():
                loss = crit(model(img_feats, labels[..., :-1]), labels[..., 1:], masks[..., 1:]).item()
            loss_sum = loss_sum + loss
            loss_evals = loss_evals + 1

        # forward the model to also get generated samples for each image
        with torch.no_grad():
            tmp_eval_kwargs = eval_kwargs.copy()
            tmp_eval_kwargs.update({'sample_n': 1})
            seq, seq_logprobs = model(img_feats, opt=tmp_eval_kwargs, mode='sample')
            seq = seq.data
            entropy = - (F.softmax(seq_logprobs, dim=2) * seq_logprobs).sum(2).sum(1) / (
                        (seq > 0).to(seq_logprobs).sum(1) + 1)
            perplexity = - seq_logprobs.gather(2, seq.unsqueeze(2)).squeeze(2).sum(1) / (
                        (seq > 0).to(seq_logprobs).sum(1) + 1)

        sents = utils.decode_sequence(model.vocab, seq)

        for k, sent in enumerate(sents):
            gt_tokens = data['labels'][k][0]
            gt_sent = [model.vocab[w.item()] for w in gt_tokens if w > 0]
            gt_sent = ' '.join(gt_sent)
            entry = {'image_id': data['infos'][k]['id'], 'caption': sent, 'gt': gt_sent, 'perplexity': perplexity[k].item(), 'entropy': entropy[k].item()}
            if eval_kwargs.get('dump_path', 0) == 1:
                entry['file_name'] = data['infos'][k]['file_path']
            predictions.append(entry)

            if verbose:
                print('image %s: %s' % (entry['image_id'], entry['caption']))

        if sample_n > 1:
            eval_split_n(model, n_predictions, [img_feats, data], eval_kwargs)
        
        # ix0 = data['bounds']['it_pos_now']
        ix1 = data['bounds']['it_max']
        if num_images != -1:
            ix1 = min(ix1, num_images)
        else:
            num_images = ix1
        for i in range(n - ix1):
            predictions.pop()

        if verbose:
            print('evaluating validation preformance... %d/%d (%f)' % (n, ix1, loss))

        if 0 <= num_images <= n:
            break

    lang_stats = None
    if len(n_predictions) > 0 and 'perplexity' in n_predictions[0]:
        n_predictions = sorted(n_predictions, key=lambda x: x['perplexity'])
    if not os.path.isdir('eval_results'):
        os.mkdir('eval_results')
    torch.save((predictions, n_predictions),
               os.path.join('eval_results/', 'saved_pred_' + eval_kwargs['id'] + '_' + split + '.pth'))
    if lang_eval == 1:
        lang_stats = language_eval(dataset, predictions, n_predictions, eval_kwargs, split)

    # Switch back to training mode
    model.train()
    return loss_sum/loss_evals, predictions, lang_stats


# Only run when sample_n > 0
def eval_split_n(model, n_predictions, input_data, eval_kwargs={}):
    verbose = eval_kwargs.get('verbose', True)
    beam_size = eval_kwargs.get('beam_size', 1)
    sample_n = eval_kwargs.get('sample_n', 1)
    sample_n_method = eval_kwargs.get('sample_n_method', 'sample')

    img_feats, data = input_data

    tmp_eval_kwargs = eval_kwargs.copy()
    if sample_n_method == 'bs':
        # case 1 sample_n == beam size
        tmp_eval_kwargs.update({'sample_n': 1, 'beam_size': sample_n, 'group_size': 1}) # randomness from softmax
        with torch.no_grad():
            model(img_feats, opt=tmp_eval_kwargs, mode='sample')
        for k in range(img_feats.shape[0]):
            _sents = utils.decode_sequence(model.vocab, torch.stack([model.done_beams[k][_]['seq'] for _ in range(sample_n)]))
            for sent in _sents:
                entry = {'image_id': data['infos'][k]['id'], 'caption': sent}
                n_predictions.append(entry)
    # case 2 sample / gumbel / topk sampling/ nucleus sampling
    elif sample_n_method == 'sample' or \
            sample_n_method == 'gumbel' or \
            sample_n_method.startswith('top'):
        tmp_eval_kwargs.update({'sample_n': sample_n, 'sample_method': sample_n_method, 'beam_size': 1}) # randomness from sample
        with torch.no_grad():
            _seq, _sampleLogprobs = model(img_feats, opt=tmp_eval_kwargs, mode='sample')
        _sents = utils.decode_sequence(model.vocab, _seq)
        _perplexity = - _sampleLogprobs.gather(2, _seq.unsqueeze(2)).squeeze(2).sum(1) / ((_seq > 0).to(_sampleLogprobs).sum(1)+1)
        for k, sent in enumerate(_sents):
            entry = {'image_id': data['infos'][k // sample_n]['id'], 'caption': sent, 'perplexity': _perplexity[k].item()}
            n_predictions.append(entry)
    else:
        tmp_eval_kwargs.update({'sample_method': sample_n_method[1:], 'group_size': sample_n, 'beam_size':1}) # randomness from softmax
        with torch.no_grad():
            _seq, _sampleLogprobs = model(img_feats, opt=tmp_eval_kwargs, mode='sample')
        _sents = utils.decode_sequence(model.vocab, _seq)
        for k, sent in enumerate(_sents):
            entry = {'image_id': data['infos'][k // sample_n]['id'], 'caption': sent}
            n_predictions.append(entry)
    if verbose:
        for entry in sorted(n_predictions[-img_feats.shape[0] * sample_n:], key=lambda x: x['image_id']):
            print('image %s: %s' %(entry['image_id'], entry['caption']))