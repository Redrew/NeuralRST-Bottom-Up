import sys, time
import numpy as np
import random
from datetime import datetime

sys.path.append(".")

import argparse
import torch

from in_out.reader import Reader
from in_out.util import load_embedding_dict, get_logger
from in_out.preprocess import create_alphabet, construct_embedding
from in_out.preprocess import batch_data_variable
from models.vocab import Vocab
from models.metric import Metric
from models.config import load_config
from train_rst_parser import predict, load_word_embedding_and_tokenizer

def evaluate_train_data(network, train_instances, vocab, config, logger):
    total_data = len(train_instances)

    def get_subtrees(data, indices):
        subtrees = []
        for i in indices:
            subtrees.append(data[i].result)
        return subtrees

    permutation = torch.randperm(total_data).long()
    network.metric_span.reset()
    network.metric_nuclear_relation.reset()
    time_start = datetime.now()
    for i in range(0, total_data, config.batch_size):
        network.train()
        network.training = True
            
        indices = permutation[i: i+config.batch_size]
        subset_data = batch_data_variable(train_instances, indices, vocab, config)
        gold_subtrees = get_subtrees(train_instances, indices)
            
        cost, cost_val = network.loss(subset_data, gold_subtrees)

        time_elapsed = datetime.now() - time_start
        m,s = divmod(time_elapsed.seconds, 60)

    logger.info('CorrectSpan: %.2f, CorrectNuclearRelation: %.2f - {} mins {} secs'.format(m,s) % (
        network.metric_span.get_accuracy(), network.metric_nuclear_relation.get_accuracy()))

def main():
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('config_path')
    args_parser.add_argument('--merge_inference', default=None)
    args = args_parser.parse_args()
    config = load_config(args.config_path)
    if args.merge_inference is not None:
        config.set_merge_selection_for_inference(args.merge_inference)
    
    logger = get_logger("RSTParser (Top-Down) RUN", config.use_dynamic_oracle, config.model_path)
    word_alpha, tag_alpha, gold_action_alpha, action_label_alpha, relation_alpha, nuclear_alpha, nuclear_relation_alpha, etype_alpha = create_alphabet(None, config.alphabet_path, logger)
    vocab = Vocab(word_alpha, tag_alpha, etype_alpha, gold_action_alpha, action_label_alpha, relation_alpha, nuclear_alpha, nuclear_relation_alpha)
    
    tag_embedd = construct_embedding(tag_alpha, config.tag_dim, config.freeze)
    etype_embedd = construct_embedding(etype_alpha, config.etype_dim, config.freeze)
    word_embedd, word_tokenizer = load_word_embedding_and_tokenizer(word_alpha, config)

    network = config.architecture_class(vocab, config, word_embedd, tag_embedd, etype_embedd) 
    network.load_state_dict(torch.load(config.model_name))

    if config.use_gpu:
        network = network.cuda()
    network.eval()
    
    network.training = False

    network.metric_span.reset()
    network.metric_nuclear_relation.reset()
    logger.info('Reading dev instance, and predict...')
    reader = Reader(config.dev_path, config.dev_syn_feat_path)
    dev_instances  = reader.read_data()
    predict(network, dev_instances, vocab, config, logger)
    logger.info('CorrectSpan: %.2f, CorrectNuclearRelation: %.2f' % (
        network.metric_span.get_accuracy(), network.metric_nuclear_relation.get_accuracy()))

    network.metric_span.reset()
    network.metric_nuclear_relation.reset()
    logger.info('Reading test instance, and predict...')
    reader = Reader(config.test_path, config.test_syn_feat_path)
    test_instances  = reader.read_data()
    predict(network, test_instances, vocab, config, logger)
    logger.info('CorrectSpan: %.2f, CorrectNuclearRelation: %.2f' % (
        network.metric_span.get_accuracy(), network.metric_nuclear_relation.get_accuracy()))

if __name__ == '__main__':
    main()
