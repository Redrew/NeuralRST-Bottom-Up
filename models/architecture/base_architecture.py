__author__ = 'Fajri Koto and Andrew Shen'

import operator
import numpy as np
import torch
import torch.nn as nn
from models.metric import Metric
from modules.layer import *

class BaseArchitecture(nn.Module):
    def __init__(self, vocab, config, word_embedd, tag_embedd, etype_embedd):
        super(BaseArchitecture, self).__init__()
        
        self.word_embedd = word_embedd
        self.tag_embedd = tag_embedd
        self.etype_embedd = etype_embedd
        
        self.config = config
        self.vocab = vocab
        self.static_embedd = config.static_word_embedding

        dim_enc1 = config.word_dim
        if self.static_embedd:
            dim_enc1 += config.tag_dim
        dim_enc2 = config.syntax_dim
        dim_enc3 = config.hidden_size * 4 + config.etype_dim

        self.rnn_word = MyLSTM(input_size=dim_enc1, hidden_size=config.hidden_size, num_layers=config.num_layers, batch_first=True, bidirectional=True, dropout_in=config.drop_prob, dropout_out=config.drop_prob)
        self.rnn_syntax = MyLSTM(input_size=dim_enc2, hidden_size=config.hidden_size, num_layers=config.num_layers, batch_first=True, bidirectional=True, dropout_in=config.drop_prob, dropout_out=config.drop_prob)
        self.rnn_edu = MyLSTM(input_size=dim_enc3, hidden_size=config.hidden_size, num_layers=config.num_layers, batch_first=True, bidirectional=True, dropout_in=config.drop_prob, dropout_out=config.drop_prob)

        self.dropout_in = nn.Dropout(p=config.drop_prob)
        self.dropout_out = nn.Dropout(p=config.drop_prob)
        out_dim1 = config.hidden_size * 2
        out_dim2 = config.hidden_size * 2
        
        self.rnn_segmentation = MyLSTM(input_size=out_dim2, hidden_size=config.hidden_size_tagger, num_layers=config.num_layers, batch_first=True, bidirectional=True, dropout_in=config.drop_prob, dropout_out=config.drop_prob)
        self.mlp_seg = NonLinear(config.hidden_size_tagger * 2, config.hidden_size_tagger/2, activation=nn.Tanh())
        
        self.mlp_nuclear_relation = NonLinear(config.hidden_size_tagger * 4, config.hidden_size, activation=nn.Tanh())
        self.output_nuclear_relation = nn.Linear(config.hidden_size, vocab.nuclear_relation_alpha.size())

        self.metric_span = Metric()
        self.metric_nuclear_relation = Metric()
        
        self.index_output = []
        for idx in range(config.batch_size):
            self.index_output.append([])

        self.training = True
        self.epoch = 0

    def run_rnn_word(self, tensor, mask):
        batch_size, edu_size, word_in_edu, hidden_size = tensor.shape
        tensor = tensor.view(batch_size * edu_size, word_in_edu, hidden_size)
        mask = mask.view(batch_size * edu_size, word_in_edu)
        tensor, hn = self.rnn_word(tensor, mask, None)
        
        tensor = tensor.transpose(0,1).contiguous()
        tensor = tensor.view(batch_size, edu_size, word_in_edu, -1)
        # tensor = self.dropout_out(tensor)
        return tensor

    def run_rnn_word_tag(self, input_word, input_tag, word_mask):
        word = self.word_embedd(input_word)
        tag = self.tag_embedd(input_tag)
        word = self.dropout_in(word)
        tag = self.dropout_in(tag)
        
        tensor = torch.cat([word, tag], dim=-1)
        # apply rnn over EDU
        tensor = self.run_rnn_word(tensor, word_mask)
        return tensor
    
    def run_rnn_token(self, input_token, token_mask):
        tensor = self.word_embedd(input_token, token_mask)
        tensor = self.dropout_in(tensor)
        
        # apply rnn over EDU
        tensor = self.run_rnn_word(tensor, token_mask)
        return tensor


    def run_rnn_syntax(self, syntax, word_mask):
        syntax = self.dropout_in(syntax)
        
        # apply rnn over EDU
        batch_size, edu_size, word_in_edu, hidden_size = syntax.shape
        syntax = syntax.view(batch_size * edu_size, word_in_edu, hidden_size)
        word_mask = word_mask.view(batch_size * edu_size, word_in_edu)
        tensor, hn = self.rnn_syntax(syntax, word_mask, None)
        
        tensor = tensor.transpose(0,1).contiguous()
        tensor = tensor.view(batch_size, edu_size, word_in_edu, -1)
        # tensor = self.dropout_out(tensor)
        return tensor
    
    def run_rnn_edu(self, word_representation, syntax_representation, token_denominator, word_denominator, input_etype, edu_mask):
        etype = self.etype_embedd(input_etype)
        etype = self.dropout_in(etype)
       
        # apply average pooling based on EDU span
        batch_size, edu_size, token_in_edu, hidden_size = word_representation.shape
        word_representation = word_representation.view(batch_size * edu_size, token_in_edu, -1)
        edu_representation1 = AvgPooling(word_representation, token_denominator.view(-1))
        edu_representation1 = edu_representation1.view(batch_size, edu_size, -1)
        
        batch_size, edu_size, word_in_edu, hidden_size = syntax_representation.shape
        syntax_representation = syntax_representation.view(batch_size * edu_size, word_in_edu, -1)
        edu_representation2 = AvgPooling(syntax_representation, word_denominator.view(-1))
        edu_representation2 = edu_representation2.view(batch_size, edu_size, -1)

        edu_representation = torch.cat([edu_representation1, edu_representation2, etype], dim=-1)
        output, hn = self.rnn_edu(edu_representation, edu_mask, None)
        output = output.transpose(0,1).contiguous()
        # output = self.dropout_out(output)
        return output

    def run_rnn_segmentation(self, segmented_encoder, segment_mask):
        batch_size, edu_size, hidden_size = segmented_encoder.shape
        output, hn = self.rnn_segmentation(segmented_encoder, segment_mask, None)
        output = output.transpose(0,1).contiguous()
        output = self.dropout_out(output)
        
        sent_scores = torch.sum(self.mlp_seg(output), dim=-1).view(batch_size, edu_size)
        sent_scores = torch.sigmoid(sent_scores)
        sent_scores = sent_scores.clone() * segment_mask
        return sent_scores, output * segment_mask.unsqueeze(2)

    def forward_all(self, input_word, input_tag, input_etype, edu_mask, token_mask, word_mask, token_denominator, word_denominator, syntax):
        if self.static_embedd:
            word_output = self.run_rnn_word_tag(input_word, input_tag, word_mask)
        else:
            word_output = self.run_rnn_token(input_word, token_mask)
        syntax_output = self.run_rnn_syntax(syntax, word_mask)
        tensor = self.run_rnn_edu(word_output, syntax_output, token_denominator, word_denominator, input_etype, edu_mask) 
        return tensor

    def update_eval_metric(self, gold_span_index, nuclear_relation, gold_nuclear_relation, len_golds):
        batch_size, _, _ = nuclear_relation.shape
        _, nuclear_relation_idx = torch.max(nuclear_relation, 2)
        
        for idx in range(batch_size):
            self.metric_span.overall_label_count += len_golds[idx]
            self.metric_nuclear_relation.overall_label_count += len_golds[idx]
            
            for idy in range(len_golds[idx]):
                span_index = self.index_output[idx][idy]
                if gold_span_index.dim() == 2:
                    self.metric_span.correct_label_count += span_index == int(gold_span_index[idx, idy])
                else:
                    self.metric_span.correct_label_count += gold_span_index[idx, idy, span_index]
                
                if nuclear_relation_idx[idx, idy].item() == gold_nuclear_relation[idx, idy].item():
                    self.metric_nuclear_relation.correct_label_count += 1

    # Primary function
    def loss(self, subset_data, gold_subtrees, epoch=0):
        raise NotImplementedError()
