__author__ = 'Fajri Koto and Andrew Shen'

import operator
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .base_architecture import BaseArchitecture
from in_out.instance import Gold, GoldBottomUp
from modules.layer import *
from typing import List
from functools import lru_cache

class MergeOrder:
    def __init__(self, option):
        if option not in ['left', 'random']:
            raise NotImplementedError(f'Merge order {option} is not implemented')
        self.option = option

    def select(self, items):
        if self.option == 'left':
            return items[0]
        elif self.option == 'random':
            return np.random.choice(items)

class SubtreeGenerator:
    def __init__(self, merge_order, target_merges, vocab):
        self.merge_order = merge_order
        self.target_merges = target_merges
        self.vocab = vocab

    def __call__(self, gold: GoldBottomUp):
        target_merges = []
        state_spans = []
        merge_idxs = []
        nuclear_relations = []
        depth = []
        while not gold.done:
            state_spans.append(gold.state_spans)
            depth.append(gold.depths)
            merge_idx = self.merge_order.select(gold.merge_idxs)
            target_merge = self.target_merges(gold.merge_mask, merge_idx)
            target_merges.append(target_merge)
            nuclear_relation = gold.nuclear_relation[merge_idx]
            nuclear_relation = self.vocab.nuclear_relation_alpha.word2id(nuclear_relation)
            merge_idxs.append(merge_idx)
            nuclear_relations.append(nuclear_relation)
            gold = gold.merge(merge_idx, "", "")
        return target_merges, state_spans, merge_idxs, nuclear_relations, depth

class TargetMerge:
    def __init__(self, option):
        if option not in ['all', 'single']:
            raise NotImplementedError(f'Target merge {option} is not implemented')
        self.option = option

    def __call__(self, merge_mask, merge_idx):
        if self.option == 'all':
            return merge_mask
        elif self.option == 'single':
            target_merge = [0 for _ in merge_mask]
            target_merge[merge_idx] = 1
            return target_merge

class SelectMerge:
    def __init__(self, option, merge_order=None, threshold=1):
        if option not in ['max', 'threshold']:
            raise NotImplementedError(f'Merge inference {option} is not implemented')
        if option == 'threshold' and merge_order is None:
            raise Exception(f'Merge order not provided with thresholded merge inference')
        self.merge_order = merge_order
        self.option = option
        self.threshold = threshold

    def __call__(self, merge_out):
        if self.option == 'max':
            return torch.argmax(merge_out)
        elif self.option == 'threshold':
            valid_merge = merge_out > self.threshold
            if not torch.any(valid_merge):
                return torch.argmax(merge_out)
            else:
                valid_merge_idxs = valid_merge.nonzero()
                idx = self.merge_order.select(range(valid_merge_idxs.shape[0]))
                return valid_merge_idxs[idx]

def get_num_other_elem(state):
    num_edus = state[-1][-1] + 1
    num_other_elem = [num_edus - (r-l+1) for l, r in state]
    return num_other_elem

class BottomUpArchitecture(BaseArchitecture):
    def __init__(self, vocab, config, word_embedd, tag_embedd, etype_embedd):
        super(BottomUpArchitecture, self).__init__(vocab, config, word_embedd, tag_embedd, etype_embedd)

        self.merge_order = MergeOrder(config.subtree_order_for_training)
        self.get_target_merge = TargetMerge(config.target_merges)
        self.generate_merge_states = SubtreeGenerator(self.merge_order, self.get_target_merge, vocab)
        self.select_merge = SelectMerge(config.merge_selection_for_inference, self.merge_order, config.merge_selection_threshold)

    def get_prediction(self, bottom_up_trees):
        gs = [Gold(bu.get_tree()) for bu in bottom_up_trees]
        subtrees = [g.get_subtree() for g in gs]
        return gs, subtrees

    def compute_loss(self, merge_outputs, target_merges, num_elems, state_masks, nuclear_relation, gold_nuclear_relation, len_golds, depths):
        batch_size, nuc_len, nuc_num = nuclear_relation.shape
        idx_ignore_nuc = self.vocab.nuclear_relation_alpha.size()
        nuc_rel_loss = F.cross_entropy(nuclear_relation.view(batch_size * nuc_len, nuc_num),
                        gold_nuclear_relation[:,:nuc_len].contiguous().view(batch_size * nuc_len),
                        ignore_index = idx_ignore_nuc)
        seg_loss = []
        state_masks = state_masks.bool()
        state_size = state_masks.shape[0]
        for idx in range(state_size):
            stack_mask = state_masks[idx]
            if not torch.any(stack_mask):
                continue
            output, target = merge_outputs[idx, stack_mask], target_merges[idx, stack_mask]
            loss_multiplier = 0
            if self.config.depth_alpha != 0:
                loss_multiplier += depths[idx][stack_mask] ** self.config.depth_alpha
            if self.config.elem_alpha != 0:
                loss_multiplier += num_elems[idx][stack_mask] ** self.config.elem_alpha
            if self.config.depth_alpha == 0 and self.config.elem_alpha == 0:
                loss_multiplier = 1
            cur_loss = (loss_multiplier * F.binary_cross_entropy(output, target, reduction="none")).sum()
            seg_loss.append(cur_loss)
        seg_loss = sum(seg_loss) / state_masks.sum()
        gold_merges = target_merges.view(batch_size, state_size//batch_size, -1)
        self.update_eval_metric(gold_merges, nuclear_relation, gold_nuclear_relation, len_golds)

        if self.config.activate_seg_loss > 0 and self.config.activate_nuc_rel_loss > 0:
            raise ValueError('at least there is an activatation of loss started in epoch 0')
        loss = 0
        if self.config.activate_seg_loss <= self.epoch:
            loss += self.config.loss_seg * seg_loss
        if self.config.activate_nuc_rel_loss <= self.epoch:
            loss += self.config.loss_nuc_rel * nuc_rel_loss
        return loss


    # --------------------------------------------------------------------------------------
    # Functions for testing start from here
    def prepare_prediction_for_testing(self, encoder_output, merge_idxs):
        batch_size, _, hidden = encoder_output.shape

        stack_state = Variable(torch.zeros(batch_size, 2, hidden))
        if self.config.use_gpu:
            stack_state = stack_state.cuda()
        for idx in range(batch_size):
            merge_idx = merge_idxs[idx]
            stack_state[idx, 0] = encoder_output[idx, merge_idx]
            stack_state[idx, 1] = encoder_output[idx, merge_idx+1]
        
        stack_state = stack_state.view(batch_size, -1)
        return stack_state

    def prepare_merge_for_testing(self, encoder_output, state_batch):
        batch_size, edu_num, hidden = encoder_output.shape
        stack_state = Variable(torch.zeros(batch_size, edu_num, hidden)).type(torch.FloatTensor)
        stack_mask = Variable(torch.zeros(batch_size, edu_num), requires_grad=False).type(torch.LongTensor)
        for idx in range(batch_size):
            state_spans = state_batch[idx]
            for idz, state_span in enumerate(state_spans):
                start, end = state_span
                stack_state[idx, idz] = encoder_output[idx, start:end+1].mean(0)
                stack_mask[idx, idz] = 1
        if self.config.use_gpu:
            stack_state = stack_state.cuda()
            stack_mask = stack_mask.cuda()
        return stack_state, stack_mask

    def decode_testing(self, encoder_output, gold_bottom_up):
        batch_size, _, hidden_size = encoder_output.shape
        state, trees = self.get_initial_state(gold_bottom_up)
        for idx in range(batch_size):
            self.index_output[idx]=[]
        
        while self.not_finished(trees):
            hidden_state1, state_mask = self.prepare_merge_for_testing(encoder_output, state)
            merge_output, rnn_output = self.run_rnn_span(hidden_state1, state_mask) #output in cuda-2
            merges = self.select_merges(merge_output, state_mask, trees)
            hidden_state2 = self.prepare_prediction_for_testing(rnn_output, merges)
            nuclear_relation_output = self.output_nuclear_relation(self.mlp_nuclear_relation(hidden_state2))
            state, trees = self.update_state(merges, nuclear_relation_output, trees)

        return self.get_prediction(trees)

    # Helper function
    def get_initial_state(self, gold_bottom_up):
        state_spans = [gold.state_spans for gold in gold_bottom_up]
        trees = [gold.make_initial_forest() for gold in gold_bottom_up]
        return state_spans, trees

    # Helper function
    def not_finished(self, trees):
        for tree in trees:
            if not tree.done:
                return True
        return False

    def select_merges(self, merge_output, state_mask, trees):
        merges = []
        batch_size = merge_output.shape[0]
        for idx in range(batch_size):
            merge_out, mask, tree = merge_output[idx], state_mask[idx], trees[idx]
            if tree.done:
                merges.append(-1)
            else:
                num_tokens = mask.sum()
                merge_idx = self.select_merge(merge_out[:num_tokens-1]).item()
                merges.append(merge_idx)
        return merges

    # Helper function
    def update_state(self, merges, nuclear_relation_output, trees):
        new_trees = []
        batch_size = len(merges)
        for idx in range(batch_size):
            merge_idx, nuclear_relation_out, tree = merges[idx], nuclear_relation_output[idx], trees[idx]
            if tree.done:
                new_trees.append(tree)
            else:
                nuclear_relation_idx = torch.argmax(nuclear_relation_out).item()
                nuclear, relation = self.vocab.nuclear_relation_alpha.id2word(nuclear_relation_idx).split(' - ')
                new_trees.append(tree.merge(merge_idx, nuclear, relation))
        state_spans = [t.state_spans for t in new_trees]
        return state_spans, new_trees
    # End of testing -----------------------------------------------------------------------


    # --------------------------------------------------------------------------------------
    # Functions for training with static oracle (normal training) start from here
    def prepare_prediction_for_training(self, encoder_output, merge_idx_batch):
        stack_size, edu_num, hidden = encoder_output.shape
        batch_size = stack_size // (edu_num - 1)

        stack_state = Variable(torch.zeros(stack_size, 2, hidden))
        if self.config.use_gpu:
            stack_state = stack_state.cuda()
        for idx in range(batch_size):
            merge_idxs = merge_idx_batch[idx]
            for idy in range(len(merge_idxs)):
                merge_idx = merge_idxs[idy]
                idz = idx * (edu_num - 1) + idy
                stack_state[idz, 0] = encoder_output[idz, merge_idx]
                stack_state[idz, 1] = encoder_output[idz, merge_idx+1]
        
        stack_state = stack_state.view(stack_size, -1)
        return stack_state

    def prepare_merge_for_training(self, encoder_output, target_merges, state_batch):
        batch_size, edu_num, hidden = encoder_output.shape
        stack_state = Variable(torch.zeros(batch_size, edu_num-1, edu_num, hidden)).type(torch.FloatTensor)
        stack_mask = Variable(torch.zeros(batch_size, edu_num-1, edu_num), requires_grad=False).type(torch.LongTensor)
        target_merge = Variable(torch.zeros(batch_size, edu_num-1, edu_num), requires_grad=False).type(torch.FloatTensor)
        @lru_cache(maxsize=None)
        def encoder_output_lookup(idx, start, end):
            return encoder_output[idx, start:end+1].mean(0)

        for idx in range(batch_size):
            merges, states = target_merges[idx], state_batch[idx]
            for idy in range(len(merges)):
                merge, state_spans = merges[idy], states[idy]
                encoder_state = []
                for state_span in state_spans:
                    start, end = state_span
                    encoder_state.append(encoder_output_lookup(idx, start, end))
                idz = len(state_spans)
                stack_state[idx, idy, :idz] = torch.stack(encoder_state)
                stack_mask[idx, idy, :idz] = 1
                idz = len(merge)
                target_merge[idx, idy, :idz] = torch.FloatTensor(merge)

        stack_size = batch_size * (edu_num - 1)
        if self.config.use_gpu:
            stack_state = stack_state.cuda()
            stack_mask = stack_mask.cuda()
            target_merge = target_merge.cuda()
        return stack_state.view(stack_size, edu_num, -1), stack_mask.view(stack_size, -1), target_merge.view(stack_size, -1)

    def set_merge_prediction_for_training(self, segment_outputs, segment_masks):
        batch_size, iters, edu_num = segment_outputs.shape
        segment_masks = torch.sum(segment_masks, dim=-1)
        assert iters == edu_num -1
        
        for idx in range(batch_size):
            for idy in range(iters):
                num_tokens = int(segment_masks[idx, idy].item())
                if num_tokens == 0:
                    continue
                out = self.select_merge(segment_outputs[idx, idy, :num_tokens-1]).item()
                self.index_output[idx].append(out)

    def decode_training(self, encoder_output, gold_bottom_up: List[GoldBottomUp], len_golds):
        batch_size, edu_num, hidden_size = encoder_output.shape
        target_merge_batch, state_batch, merge_idx_batch = [], [], []
        depth_batch = Variable(torch.zeros(batch_size, edu_num-1, edu_num).type(torch.LongTensor), requires_grad=False)
        num_other_elems = Variable(torch.zeros(batch_size, edu_num-1, edu_num).type(torch.LongTensor), requires_grad=False)
        gold_nuclear_relation = Variable(torch.ones(batch_size, edu_num-1).type(torch.LongTensor) * self.vocab.nuclear_relation_alpha.size(), requires_grad=False)
        for idx in range(batch_size):
            self.index_output[idx] = []
            merges, states, merge_idxs, nuclear_relations, depths = self.generate_merge_states(gold_bottom_up[idx])
            target_merge_batch.append(merges)
            state_batch.append(states)
            merge_idx_batch.append(merge_idxs)
            num_states = len(states)
            for idy in range(num_states):
                num_merges = len(states[idy])
                depth_batch[idx, idy, :num_merges] = torch.tensor(depths[idy])
                num_other_elems[idx, idy, :num_merges] = torch.tensor(get_num_other_elem(states[idy]))
            gold_nuclear_relation[idx, :num_states] = torch.tensor(nuclear_relations)
        if self.config.use_gpu:
            depth_batch = depth_batch.cuda()
            gold_nuclear_relation = gold_nuclear_relation.cuda()
            num_other_elems = num_other_elems.cuda()
        depth_batch = depth_batch.view(batch_size * (edu_num-1), -1)
        num_other_elems = num_other_elems.view(batch_size * (edu_num-1), -1)

        all_hidden_states1, state_masks, target_merges = self.prepare_merge_for_training(encoder_output, target_merge_batch, state_batch)
        merge_outputs, rnn_outputs = self.run_rnn_span(all_hidden_states1, state_masks)
        self.set_merge_prediction_for_training(merge_outputs.view(batch_size, edu_num-1, -1), 
                state_masks.view(batch_size, edu_num-1, -1))
        all_hidden_states2 = self.prepare_prediction_for_training(rnn_outputs, merge_idx_batch)
        nuclear_relation_outputs = self.output_nuclear_relation(self.mlp_nuclear_relation(all_hidden_states2))
        nuclear_relation_outputs = nuclear_relation_outputs.view(batch_size, edu_num-1, -1)
        return self.compute_loss(merge_outputs, target_merges, num_other_elems, state_masks, \
            nuclear_relation_outputs, gold_nuclear_relation, len_golds, depth_batch)
    # End of training with static oracle ---------------------------------------------------


    # --------------------------------------------------------------------------------------
    # Functions for training with dynamic oracle starts from here
    def get_initial_state_oracle(self, gold_bottom_up):
        state_spans = [gold.state_spans for gold in gold_bottom_up]
        return state_spans, gold_bottom_up

    def select_merges_oracle(self, merge_output, state_mask, trees):
        merges, merges_gold = [], []
        batch_size = merge_output.shape[0]
        for idx in range(batch_size):
            p = random.random()
            merge_out, mask, tree = merge_output[idx], state_mask[idx], trees[idx]
            if tree.done:
                merges.append(-1)
                merges_gold.append(-1)
            else:
                num_tokens = mask.sum()
                gold_merge_idx = self.merge_order.select(tree.merge_idxs)
                if p > self.config.oracle_prob:
                    merge_idx = gold_merge_idx
                else:
                    merge_idx = self.select_merge(merge_out[:num_tokens-1]).item()
                self.index_output[idx].append(merge_idx)
                merges.append(merge_idx)
                merges_gold.append(gold_merge_idx)
        return merges, merges_gold

    def update_state_oracle(self, merges, merges_gold, trees, edu_num):
        new_trees = []
        batch_size = len(merges)
        merge_gold = Variable(torch.zeros(batch_size, edu_num).type(torch.FloatTensor), requires_grad=False)
        nuclear_relation_gold = Variable(torch.ones(batch_size, 1).type(torch.LongTensor) * self.vocab.nuclear_relation_alpha.size(), requires_grad=False)
        depths = Variable(torch.zeros(batch_size, edu_num).type(torch.LongTensor), requires_grad=False)
        num_other_elems = Variable(torch.zeros(batch_size, edu_num).type(torch.LongTensor), requires_grad=False)
        for idx in range(batch_size):
            merge_idx, gold_merge_idx, tree = merges[idx], merges_gold[idx], trees[idx]
            if tree.done:
                new_trees.append(tree)
            else:
                state_size = len(tree.state_spans)
                new_trees.append(tree.merge(merge_idx, "", ""))
                target_merge = self.get_target_merge(tree.merge_mask, gold_merge_idx)
                merge_gold[idx, :len(target_merge)] = torch.tensor(target_merge)
                nuclear_relation = tree.nuclear_relation[merge_idx]
                nuclear_relation = self.vocab.nuclear_relation_alpha.word2id(nuclear_relation)
                nuclear_relation_gold[idx] = nuclear_relation
                depths[idx, :state_size] = torch.tensor(tree.depths)
                num_other_elems[idx, :state_size] = torch.tensor(get_num_other_elem(tree.state_spans))

        if self.config.use_gpu:
            merge_gold = merge_gold.cuda()
            nuclear_relation_gold = nuclear_relation_gold.cuda()
            depths = depths.cuda()
            num_other_elems = num_other_elems.cuda()

        state_spans = [t.state_spans for t in new_trees]
        return state_spans, new_trees, merge_gold, nuclear_relation_gold, depths, num_other_elems


    def decode_training_dynamic_oracle(self, encoder_output, gold_bottom_up: List[GoldBottomUp], len_golds):
        batch_size, edu_num, hidden_size = encoder_output.shape
        state, trees = self.get_initial_state_oracle(gold_bottom_up)
        for idx in range(batch_size):
            self.index_output[idx]=[]
        merge_outputs, target_merges, state_masks = [], [], []
        nuclear_relation_outputs, all_nuclear_relation_gold = [], []
        all_depths, num_other_elems = [], []
        
        while self.not_finished(trees):
            hidden_state1, state_mask = self.prepare_merge_for_testing(encoder_output, state)
            merge_output, rnn_output = self.run_rnn_span(hidden_state1, state_mask) #output in cuda-2

            merges, merges_gold = self.select_merges_oracle(merge_output, state_mask, trees)
            hidden_state2 = self.prepare_prediction_for_testing(rnn_output, merges)
            nuclear_relation_output = self.output_nuclear_relation(self.mlp_nuclear_relation(hidden_state2))

            # Update trees, get gold merge, gold nuclear_relation
            state, trees, target_merge, nuclear_relation_gold, depths, other_elems = self.update_state_oracle(merges, merges_gold, trees, edu_num)
            merge_outputs.append(merge_output)
            state_masks.append(state_mask)
            target_merges.append(target_merge)
            all_depths.append(depths)
            num_other_elems.append(other_elems)
            nuclear_relation_outputs.append(nuclear_relation_output.view(batch_size, 1, -1))
            all_nuclear_relation_gold.append(nuclear_relation_gold.view(batch_size, 1, -1))

        merge_outputs = torch.cat(merge_outputs, dim=0)
        target_merges = torch.cat(target_merges, dim=0)
        state_masks = torch.cat(state_masks, dim=0)
        all_depths = torch.cat(all_depths, dim=0)
        num_other_elems = torch.cat(num_other_elems, dim=0)
        nuclear_relation_outputs = torch.cat(nuclear_relation_outputs, dim=1)
        all_nuclear_relation_gold = torch.cat(all_nuclear_relation_gold, dim=1)

        return self.compute_loss(merge_outputs, target_merges, num_other_elems, state_masks, \
            nuclear_relation_outputs, all_nuclear_relation_gold, len_golds, all_depths)
    # End of training with DYNAMIC oracle ---------------------------------------------------


    # Primary function
    def loss(self, subset_data, gold_subtrees, epoch=0):
        self.epoch = epoch
        words, tags, etypes, edu_mask, token_mask, word_mask, len_edus, token_denominator, word_denominator, syntax, \
            gold_nuclear, gold_relation, gold_nuclear_relation, gold_segmentation, span, len_golds, depth, gold_bottom_up = subset_data
        encoder_output = self.forward_all(words, tags, etypes, edu_mask, token_mask, word_mask, token_denominator, word_denominator, syntax)
        if self.training:
            if self.config.flag_oracle:
                cost = self.decode_training_dynamic_oracle(encoder_output, gold_bottom_up, len_golds)
            else:
                cost = self.decode_training(encoder_output, gold_bottom_up, len_golds)
            return cost, cost.item()
        else:
            if self.config.beam_search == 1:
                gs, results = self.decode_testing(encoder_output, gold_bottom_up)
            else: #do beam search
                # not supported yet
                raise NotImplementedError('Beam search has not been implemented')
            return gs, results