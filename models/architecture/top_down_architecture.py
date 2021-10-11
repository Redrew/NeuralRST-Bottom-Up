__author__ = 'Fajri Koto and Andrew Shen'

import copy
import operator
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .base_architecture import BaseArchitecture
from in_out.instance import Gold
from modules.layer import *

class TopDownArchitecture(BaseArchitecture):
    def get_prediction(self, all_nuclear_relation_output):
        batch_size, _, _ = all_nuclear_relation_output.shape
        _, nuclear_relation_idx = torch.max(all_nuclear_relation_output, 2)
       
        gs = []
        subtrees = []
        for idx in range(batch_size):
            span = self.all_span_output[idx]
            segment_cut = self.index_output[idx]
            segment_cut = segment_cut[:len(span)]
            segment = []
            nuclear = []
            relation = []
            for idy in range(len(span)):
                segment.append(span[idy][0]+segment_cut[idy])
                nuclear_relation = self.vocab.nuclear_relation_alpha.id2word(nuclear_relation_idx[idx, idy].item()).split(' - ')
                nuclear.append(nuclear_relation[0])
                relation.append(nuclear_relation[1])
            g = Gold(None)
            gs.append(g)
            g.set_attribute(span, segment, nuclear, relation)
            subtrees.append(g.get_subtree())
        return gs, subtrees

    def compute_loss(self, segmentation, gold_segmentation, segment_mask, nuclear_relation, gold_nuclear_relation, len_golds, depth):    
        #nuclear_relation loss
        batch_size, nuc_len, nuc_num = nuclear_relation.shape
        idx_ignore_nuc = self.vocab.nuclear_relation_alpha.size()
        nuc_rel_loss = F.cross_entropy(nuclear_relation.view(batch_size * nuc_len, nuc_num),
                        gold_nuclear_relation[:,:nuc_len].contiguous().view(batch_size * nuc_len),
                        ignore_index = idx_ignore_nuc)

        #segmentation loss
        batch_size, segment_len, segment_num = segmentation.shape
        val, gold_segmentation_index = gold_segmentation.max(2)
        seg_loss = []
        for idx in range(batch_size):
            if len_golds[idx] > 0:
                for idy in range(len_golds[idx]):
                    cur_segment = segmentation[idx, idy]
                    cur_segment_gold = gold_segmentation[idx, idy]
                    loss_multiplier = (1/depth[idx][idy]**self.config.depth_alpha) + (segment_mask[idx][idy].sum()**self.config.elem_alpha)
                    if self.config.depth_alpha == 0 and self.config.elem_alpha == 0:
                        loss_multiplier = 1
                    cur_loss = loss_multiplier * F.binary_cross_entropy(cur_segment, cur_segment_gold, reduction='sum')
                    seg_loss.append(cur_loss)
        seg_loss = sum(seg_loss) / segment_mask.sum()
        self.update_eval_metric(gold_segmentation_index, nuclear_relation, gold_nuclear_relation, len_golds)
        
        if self.config.activate_seg_loss > 0 and self.config.activate_nuc_rel_loss > 0:
            raise ValueError('at least there is an activatation of loss started in epoch 0')
        loss = 0
        if self.config.activate_seg_loss <= self.epoch:
            loss += self.config.loss_seg * seg_loss
        if self.config.activate_nuc_rel_loss <= self.epoch:
            loss += self.config.loss_nuc_rel * nuc_rel_loss
        return loss
    
    # Helper function
    def not_finished(self, span, batch_size):
        for idx in range(batch_size):
            if len(span[idx])>0:
                return True
        return False

    # Helper function
    def get_initial_span(self, span, batch_size):
        span_initial = []
        for idx in range(batch_size):
            if len(span[idx]) > 0:
                span_initial.append([span[idx][0]])
            else:
                span_initial.append([])
        return span_initial


    # --------------------------------------------------------------------------------------
    # Functions for testing start from here
    def update_span(self, segment_output, segment_mask, span_initial):
        batch_size, max_edu_num = segment_output.shape
        segment_mask = segment_mask.sum(-1)
        ret_val = []

        for idx in range(batch_size):
            if len(span_initial[idx]) == 0:
                ret_val.append((None, None))
                continue
            limitted_span = int(segment_mask[idx])

            _, index = segment_output[idx, :limitted_span-1].max(0)
            index = int(index)
            self.index_output[idx].append(index)

            edu_span = span_initial[idx].pop(0)
            span1 = (edu_span[0], edu_span[0]+index)
            span2 = (edu_span[0]+index+1, edu_span[1])
            if span2[0] > span2[1]:
                import ipdb; ipdb.set_trace()

            ret_val.append((span1, span2))
            if span1[0] != span1[1]:
                span_initial[idx].append(span1)
                self.all_span_output[idx].append(span1)
            if span2[0] != span2[1]:
                span_initial[idx].append(span2)
                self.all_span_output[idx].append(span2)
        return ret_val, span_initial
    
    def prepare_segmentation_for_testing(self, encoder_output, span_initial):
        batch_size, edu_num, hidden = encoder_output.shape
        bucket = Variable(torch.zeros(batch_size, 1, hidden)).type(torch.FloatTensor)
        if self.config.use_gpu:
            bucket = bucket.cuda()
        edu_rep = torch.cat((encoder_output, bucket), 1) # batch_size, action_num + 1, hidden_size
        edu_rep = edu_rep.view(batch_size * (edu_num + 1), hidden)
        
        segment_mask = Variable(torch.zeros(batch_size, edu_num)).type(torch.FloatTensor)
        stack_index = Variable(torch.ones(batch_size * edu_num)).type(torch.LongTensor) * edu_num
        for idx in range(batch_size):
            stack_offset = idx * edu_num
            value_offset = idx * (edu_num + 1)
            if len(span_initial[idx]) > 0:
                edu_span = span_initial[idx][0]
                l = edu_span[1]-edu_span[0] + 1
                for j in range(l):
                    stack_index[stack_offset + j] = value_offset + edu_span[0] + j
                    segment_mask[idx, j] = 1

        if self.config.use_gpu:
            stack_index = stack_index.cuda()
            segment_mask = segment_mask.cuda()

        stack_state = torch.index_select(edu_rep, 0, stack_index)
        stack_state = stack_state.view(batch_size, edu_num, hidden)
        return stack_state, segment_mask

    def prepare_prediction_for_testing(self, encoder_output, cur_span_pairs):
        batch_size, edu_num, hidden = encoder_output.shape
        bucket = Variable(torch.zeros(batch_size, 1, hidden)).type(torch.FloatTensor)
        if self.config.use_gpu:
            bucket = bucket.cuda()
        edu_rep = torch.cat((encoder_output, bucket), 1) # batch_size, action_num + 1, hidden_size
        edu_rep = edu_rep.view(batch_size * (edu_num + 1), hidden)
        
        stack_index = Variable(torch.ones(batch_size * 2 * edu_num)).type(torch.LongTensor) * edu_num
        stack_denominator = Variable(torch.ones(batch_size * 2)).type(torch.FloatTensor) * -1

        for idx in range(batch_size):
            s1, s2 = cur_span_pairs[idx]
            stack_offset = idx * 2 * edu_num
            value_offset = idx * (edu_num + 1)
            denominator_offset = idx * 2
            if s1 is not None:
                s1 = list(s1); s2 = list(s2)
                starting = s1[0]
                s1[0] -= starting; s1[1] -= starting
                s2[0] -= starting; s2[1] -= starting
                
                l1 = s1[1]-s1[0] + 1
                for j in range(l1):
                    stack_index[edu_num * 0 + stack_offset + j] = value_offset + s1[0] + j
                stack_denominator [denominator_offset + 0] = l1
            
            #if s2 is not None:
                l2 = s2[1]-s2[0] + 1
                for j in range(l2):
                    stack_index[edu_num * 1 + stack_offset + j] = value_offset + s2[0] + j
                stack_denominator [denominator_offset + 1] = l2
        
        if self.config.use_gpu:
            stack_index = stack_index.cuda()
            stack_denominator = stack_denominator.cuda()
        
        try:
            stack_state = torch.index_select(edu_rep, 0, stack_index)
        except:
            import ipdb; ipdb.set_trace()
        stack_state = stack_state.view(batch_size * 2, edu_num, hidden)
        stack_state = AvgPooling(stack_state, stack_denominator)
        stack_state = stack_state.view(batch_size, -1)
        return stack_state

    def decode_testing(self, encoder_output, span):
        batch_size, edu_size, hidden_size = encoder_output.shape
        span_initial = self.get_initial_span(span, batch_size) #act as queue
        for idx in range(batch_size):
            self.index_output[idx]=[]

        self.all_span_output = copy.deepcopy(span_initial)
        all_segment_output = []
        all_segment_mask = []
        all_nuclear_relation_output = []
        while (self.not_finished(span_initial, batch_size)):
            hidden_state1, segment_mask = self.prepare_segmentation_for_testing(encoder_output, span_initial)
            
            segment_output, rnn_output = self.run_rnn_span(hidden_state1, segment_mask) #output in cuda-2
            cur_span_pairs, span_initial = self.update_span(segment_output, segment_mask, span_initial)

            hidden_state2 = self.prepare_prediction_for_testing(rnn_output, cur_span_pairs)
            nuclear_relation_output = self.output_nuclear_relation(self.mlp_nuclear_relation(hidden_state2))

            all_segment_output.append(segment_output.view(batch_size, 1, -1))
            all_segment_mask.append(segment_mask.view(batch_size, 1, -1))
            all_nuclear_relation_output.append(nuclear_relation_output.view(batch_size, 1, -1))
        
        all_segment_output = torch.cat(all_segment_output, dim=1)
        all_segment_mask = torch.cat(all_segment_mask, dim=1)
        all_nuclear_relation_output = torch.cat(all_nuclear_relation_output, dim=1)
        return self.get_prediction(all_nuclear_relation_output)
    # End of testing -----------------------------------------------------------------------


    # --------------------------------------------------------------------------------------
    # Functions for training with static oracle (normal training) start from here
    def prepare_prediction_for_training(self, encoder_output, segment_mask, gold_segmentation):
        batch_size, edu_num, hidden = encoder_output.shape
        bucket = Variable(torch.zeros(batch_size, 1, hidden)).type(torch.FloatTensor)
        if self.config.use_gpu:
            bucket = bucket.cuda()
        edu_rep = torch.cat((encoder_output, bucket), 1) 
        edu_rep = edu_rep.view(batch_size * (edu_num + 1), hidden)
        
        stack_index = Variable(torch.ones(batch_size * 2 * edu_num)).type(torch.LongTensor) * edu_num
        stack_denominator = Variable(torch.ones(batch_size * 2)).type(torch.FloatTensor) * -1
        
        gold_segmentation = gold_segmentation.view(batch_size, edu_num)
        _, gold_index = gold_segmentation.max(1)
        segment_mask = torch.sum(segment_mask, dim=-1).view(batch_size)
        for idx in range(batch_size):
            if segment_mask[idx] == 0:
                continue
            s1 = [0, int(gold_index[idx].item())]
            s2 = [int(gold_index[idx].item())+1, int(segment_mask[idx].item())-1]

            stack_offset = idx * 2 * edu_num
            value_offset = idx * (edu_num + 1)
            denominator_offset = idx * 2

            l1 = s1[1]-s1[0] + 1
            for j in range(l1):
                stack_index[edu_num * 0 + stack_offset + j] = value_offset + s1[0] + j
            stack_denominator [denominator_offset + 0] = l1
        
            l2 = s2[1]-s2[0] + 1
            for j in range(l2):
                stack_index[edu_num * 1 + stack_offset + j] = value_offset + s2[0] + j
            stack_denominator [denominator_offset + 1] = l2
        
        if self.config.use_gpu:
            stack_index = stack_index.cuda()
            stack_denominator = stack_denominator.cuda()
        
        try:
            stack_state = torch.index_select(edu_rep, 0, stack_index)
        except:
            import ipdb; ipdb.set_trace()
        stack_state = stack_state.view(batch_size * 2, edu_num, hidden)
        stack_state = AvgPooling(stack_state, stack_denominator)
        stack_state = stack_state.view(batch_size, -1)
        return stack_state

    def prepare_segmentation_for_training(self, encoder_output, gold_span):
        batch_size, edu_num, hidden = encoder_output.shape
        assert batch_size == len(gold_span)

        bucket = Variable(torch.zeros(batch_size, 1, hidden)).type(torch.FloatTensor)
        if self.config.use_gpu:
            bucket = bucket.cuda()
        edu_rep = torch.cat((encoder_output, bucket), 1) # batch_size, edu_num + 1, hidden_size
        edu_rep = edu_rep.view(batch_size * (edu_num + 1), hidden)
        
        segment_mask = Variable(torch.zeros(batch_size, edu_num-1, edu_num)).type(torch.FloatTensor)
        stack_index = Variable(torch.ones(batch_size * (edu_num-1) * edu_num)).type(torch.LongTensor) * edu_num
        
        for idx in range(batch_size):
            stack_offset = idx * (edu_num-1) * edu_num
            value_offset = idx * (edu_num + 1)
            for idy in range(len(gold_span[idx])):
                cur_span = gold_span[idx][idy]
                l = cur_span[1] - cur_span[0] + 1
                for idz in range(l):
                    stack_index[stack_offset + idz] = value_offset + cur_span[0] + idz
                    segment_mask[idx, idy, idz] = 1
                stack_offset +=  (edu_num)
        
        if self.config.use_gpu:
            stack_index = stack_index.cuda()
            segment_mask = segment_mask.cuda()

        stack_state = torch.index_select(edu_rep, 0, stack_index)
        stack_state = stack_state.view(batch_size * (edu_num-1), edu_num, hidden)
        segment_mask = segment_mask.view(batch_size * (edu_num-1), edu_num)
        return stack_state, segment_mask
    
    def set_segment_prediction_for_training(self, segment_outputs, segment_masks):
        batch_size, iters, edu_num = segment_outputs.shape
        segment_masks = torch.sum(segment_masks, dim=-1)
        assert iters == edu_num -1
        
        for idx in range(batch_size):
            for idy in range(iters):
                num_tokens = int(segment_masks[idx, idy].item())
                if num_tokens == 0:
                    continue
                _, out = segment_outputs[idx, idy, :num_tokens-1].max(0)
                self.index_output[idx].append(int(out))

    # Gather all of span possibilities during training, avoid the loop
    def decode_training(self, encoder_output, gold_nuclear_relation, gold_segmentation, span, len_golds, depth):
        batch_size, edu_size, hidden_size = encoder_output.shape
        for idx in range(batch_size):
            self.index_output[idx]=[]
        
        all_hidden_states1, segment_masks = self.prepare_segmentation_for_training(encoder_output, span)
        segment_outputs, rnn_outputs = self.run_rnn_span(all_hidden_states1, segment_masks)
        
        all_hidden_states2 = self.prepare_prediction_for_training(rnn_outputs, segment_masks, gold_segmentation)
        nuclear_relation_outputs = self.output_nuclear_relation(self.mlp_nuclear_relation(all_hidden_states2))
        
        #obtain segmented_index prediction
        self.set_segment_prediction_for_training(segment_outputs.view(batch_size, edu_size-1, -1), 
                segment_masks.view(batch_size, edu_size-1, -1))

        segment_outputs = segment_outputs.view(batch_size, edu_size-1, -1)
        segment_masks = segment_masks.view(batch_size, edu_size-1, -1)
        nuclear_relation_outputs = nuclear_relation_outputs.view(batch_size, edu_size-1, -1)
        
        return self.compute_loss(segment_outputs, gold_segmentation, segment_masks, \
                                     nuclear_relation_outputs, gold_nuclear_relation, \
                                     len_golds, depth)
    # End of training with static oracle ---------------------------------------------------
 

    # --------------------------------------------------------------------------------------
    # Functions for training with dynamic oracle starts from here
    # In the paper, oracle_attr is GO = Gold Order
    def update_span_oracle(self, segment_output, segment_mask, span_initial, oracle_gold_segmentation):
        batch_size, max_edu_num = segment_output.shape
        segment_mask = segment_mask.sum(-1)
        ret_val = []
        
        p = random.random()
        for idx in range(batch_size):
            if len(span_initial[idx]) == 0:
                ret_val.append((None, None))
                continue
            limitted_span = int(segment_mask[idx])

            _, next_index_gold = oracle_gold_segmentation[idx, 0, :limitted_span-1].max(0)
            next_index_gold = int(next_index_gold.item())
            
            _, next_index_pred = segment_output[idx, :limitted_span-1].max(0)
            next_index_pred = int(next_index_pred.item())

            next_index = next_index_pred
            if p > self.config.oracle_prob: #apply oracle label
                next_index = next_index_gold

            self.index_output[idx].append(next_index_pred)
            
            # For next span
            edu_span = span_initial[idx].pop(0)
            next_span1 = (edu_span[0], edu_span[0]+next_index)
            next_span2 = (edu_span[0]+next_index+1, edu_span[1])
            if next_span2[0] > next_span2[1]:
                import ipdb; ipdb.set_trace()
            if next_span1[0] != next_span1[1]:
                span_initial[idx].append(next_span1)
            if next_span2[0] != next_span2[1]:
                span_initial[idx].append(next_span2)
            
            # For predicting discourse label, we still use gold / optimal split, provided by oracle
            gold_span1 = (edu_span[0], edu_span[0]+next_index_gold)
            gold_span2 = (edu_span[0]+next_index_gold+1, edu_span[1])
            if gold_span2[0] > gold_span2[1]:
                import ipdb; ipdb.set_trace()
            ret_val.append((gold_span1, gold_span2)) 
     
        return ret_val, span_initial
    
    def prepare_oracle(self, gold_span, gold_segmentation):
        oracle_attr = []
        batch_size = len(gold_span)
        for idx in range(batch_size):
            cur_spans = gold_span[idx]
            oracle_seg = np.ones(len(cur_spans)+1, dtype=int) * len(cur_spans)
            for idy in range(len(cur_spans)):
                _, cutting_index = gold_segmentation[idx][idy].max(0)
                cutting_index = cur_spans[idy][0] + int(cutting_index)
                oracle_seg[cutting_index] = idy
            oracle_attr.append(oracle_seg)
        return oracle_attr

    def get_oracle_gold(self, span_initial, gold_span, gold_segmentation, gold_nuclear_relation, oracle_attr ):
        batch_size, iters, max_edu_num = gold_segmentation.shape
        oracle_gold_segmentation = Variable(torch.Tensor(batch_size, 1, max_edu_num).zero_(), requires_grad=False)
        oracle_gold_nuclear_relation = Variable(torch.ones(batch_size, 1).type(torch.LongTensor) * self.vocab.nuclear_relation_alpha.size(), requires_grad=False)

        for idx in range(batch_size):
            if len(span_initial[idx]) == 0:
                # return empty/PAD gold
                continue
            cur_span = span_initial[idx][0]
            s0 = cur_span[0]; s1 = cur_span[1]
            new_index = np.argmin(oracle_attr[idx][s0:s1])
            index_to_iters = np.min(oracle_attr[idx][s0:s1])

            oracle_gold_segmentation[idx, 0, new_index] = 1
            oracle_gold_nuclear_relation[idx, 0] = gold_nuclear_relation[idx, index_to_iters] # -- will be ignored!
        return oracle_gold_segmentation, oracle_gold_nuclear_relation

    def decode_training_dynamic_oracle(self, encoder_output, gold_nuclear_relation, 
            gold_segmentation, span, len_golds, depth):
        
        batch_size, edu_size, hidden_size = encoder_output.shape
        span_initial = self.get_initial_span(span, batch_size) #act as queue
        oracle_attr = self.prepare_oracle(span, gold_segmentation)
        for idx in range(batch_size):
            self.index_output[idx]=[]
        all_segment_output = []; all_segment_mask = []; all_nuclear_relation_output = []
        all_segment_gold = []; all_nuclear_relation_gold = []
        while (self.not_finished(span_initial, batch_size)):
            hidden_state1, segment_mask = self.prepare_segmentation_for_testing(encoder_output, span_initial)
            oracle_gold_segmentation, oracle_gold_nuclear_relation = self.get_oracle_gold(span_initial, span, gold_segmentation, gold_nuclear_relation, oracle_attr)
            
            segment_output, rnn_output = self.run_rnn_span(hidden_state1, segment_mask) #output in cuda-2
            cur_span_pairs, span_initial = self.update_span_oracle(segment_output, segment_mask, span_initial, oracle_gold_segmentation)

            hidden_state2 = self.prepare_prediction_for_testing(rnn_output, cur_span_pairs)
            nuclear_relation_output = self.output_nuclear_relation(self.mlp_nuclear_relation(hidden_state2))
            
            all_segment_output.append(segment_output.view(batch_size, 1, -1))
            all_nuclear_relation_output.append(nuclear_relation_output.view(batch_size, 1, -1))
            all_segment_mask.append(segment_mask.view(batch_size, 1, -1))
            all_segment_gold.append(oracle_gold_segmentation)
            all_nuclear_relation_gold.append(oracle_gold_nuclear_relation)

        all_segment_output = torch.cat(all_segment_output, dim=1)
        all_segment_mask = torch.cat(all_segment_mask, dim=1)
        all_segment_gold = torch.cat(all_segment_gold, dim=1)
        all_nuclear_relation_output = torch.cat(all_nuclear_relation_output, dim=1)
        all_nuclear_relation_gold = torch.cat(all_nuclear_relation_gold, dim=1)
        if self.config.use_gpu:
            all_segment_gold = all_segment_gold.cuda()
            all_nuclear_relation_gold = all_nuclear_relation_gold.cuda()

        return self.compute_loss(all_segment_output, all_segment_gold, all_segment_mask, \
                                     all_nuclear_relation_output, all_nuclear_relation_gold, \
                                     len_golds, depth)
    # End of training with DYNAMIC oracle ---------------------------------------------------

    
    # Primary function
    def loss(self, subset_data, gold_subtrees, epoch=0):
        # subset_data = edu_words, edu_tags, edu_types, edu_mask, word_mask, len_edus, word_denominator, edu_syntax,
        # gold_nuclear, gold_relation, gold_segmentation, span, len_golds, depth
        self.epoch = epoch
        words, tags, etypes, edu_mask, token_mask, word_mask, len_edus, token_denominator, word_denominator, syntax, \
            gold_nuclear, gold_relation, gold_nuclear_relation, gold_segmentation, span, len_golds, depth, gold_bottom_up = subset_data
        encoder_output = self.forward_all(words, tags, etypes, edu_mask, token_mask, word_mask, token_denominator, word_denominator, syntax)
        if self.training:
            if self.config.flag_oracle:
                cost = self.decode_training_dynamic_oracle(encoder_output, gold_nuclear_relation, gold_segmentation, span, len_golds, depth)
            else:
                cost = self.decode_training(encoder_output, gold_nuclear_relation, gold_segmentation, span, len_golds, depth)
            return cost, cost.item()
        else:
            if self.config.beam_search == 1:
                gs, results = self.decode_testing(encoder_output, span)
            else: #do beam search
                # not supported yet
                raise NotImplementedError('Beam search has not been implemented')
            return gs, results
