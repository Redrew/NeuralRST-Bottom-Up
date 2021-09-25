import os
import torch
from models.architecture import get_architecture_class, BaseArchitecture

STATIC_EMBEDDINGS = ["word2vec", "glove", "senna", "sskip", "polyglot"]

class Config(object):
    def __init__(self, args):
        if args is None:
            return
        self.architecture = args.architecture
        self.use_gpu = torch.cuda.is_available()
        self.use_dynamic_oracle = args.use_dynamic_oracle == 1
        self.flag_oracle = False
        self.word_embedding = args.word_embedding
        self.word_embedding_file = args.word_embedding_file
    
        self.train_path = args.train
        self.test_path = args.test
        self.dev_path = args.dev
        self.train_syn_feat_path = args.train_syn_feat
        self.test_syn_feat_path = args.test_syn_feat
        self.dev_syn_feat_path = args.dev_syn_feat
        self.model_path = args.model_path +'/'+ args.experiment
        self.model_name = args.model_name
        self.alphabet_path = os.path.join(self.model_path, 'alphabets/')

        self.max_iter = args.max_iter
        self.word_dim = args.word_dim
        self.tag_dim = args.tag_dim
        self.etype_dim = args.etype_dim
        self.syntax_dim = args.syntax_dim
        self.max_sent_size = args.max_sent_size
        self.max_edu_size = args.max_edu_size
        self.max_state_size = args.max_state_size
        self.hidden_size = args.hidden_size
        self.hidden_size_tagger = args.hidden_size_tagger
        
        self.freeze = args.freeze
        self.drop_prob = args.drop_prob
        self.num_layers = args.num_layers

        self.batch_size = args.batch_size
        self.opt = args.opt
        self.lr = args.lr
        self.ada_eps = args.ada_eps
        self.momentum = 0.9
        self.beta1 = args.beta1
        self.beta2 = args.beta2 
        self.betas = (self.beta1, self.beta2)
        self.gamma = args.gamma
        self.start_decay = args.start_decay
        self.grad_accum = args.grad_accum

        self.clip = args.clip

        self.loss_nuc_rel = args.loss_nuc_rel
        self.loss_seg = args.loss_seg
        self.activate_nuc_rel_loss = args.activate_nuc_rel_loss
        self.activate_seg_loss = args.activate_seg_loss

        self.decay = args.decay
        self.oracle_prob = args.oracle_prob
        self.start_dynamic_oracle = args.start_dynamic_oracle
        self.early_stopping = args.early_stopping
    
        self.beam_search = args.beam_search
        self.depth_alpha = args.depth_alpha
        self.elem_alpha = args.elem_alpha
        self.seed = args.seed

    @property
    def static_word_embedding(self) -> bool:
        return self.word_embedding in STATIC_EMBEDDINGS

    @property
    def contextual_word_embedding(self) -> bool:
        return not self.static_word_embedding

    @property
    def architecture_class(self) -> BaseArchitecture:
        return get_architecture_class(self.architecture)

    def save(self):
        f = open(self.model_path + '/config.cfg', 'w')
        self.dump(f)
        f.close()

    def dump(self, file_object):
        file_object.write("architecture = " + self.architecture + '\n')
        file_object.write("use_gpu = " + str(self.use_gpu) + '\n')
        file_object.write("use_dynamic_oracle = "+ str(self.use_dynamic_oracle) + '\n')
        file_object.write("flag_oracle = " + str(self.flag_oracle) + '\n')
        file_object.write("word_embedding = " + str(self.word_embedding) + '\n')
        file_object.write("word_embedding_file = " + str(self.word_embedding_file) + '\n')
    
        file_object.write("train_path = " + str(self.train_path) + '\n')
        file_object.write("test_path = " + str(self.test_path) + '\n')
        file_object.write("dev_path = " + str(self.dev_path) + '\n')
        file_object.write("train_syn_feat_path = " + str(self.train_syn_feat_path) + '\n')
        file_object.write("test_syn_feat_path = " + str(self.test_syn_feat_path) + '\n')
        file_object.write("dev_syn_feat_path = " + str(self.dev_syn_feat_path) + '\n')
        file_object.write("model_path = " + str(self.model_path) + '\n')
        file_object.write("model_name = " + str(self.model_name) + '\n')
        file_object.write("alphabet_path = " + str(self.alphabet_path) + '\n')

        file_object.write("max_iter = " + str(self.max_iter) + '\n')
        file_object.write("word_dim = " + str(self.word_dim) + '\n')
        file_object.write("tag_dim = " + str(self.tag_dim) + '\n')
        file_object.write("etype_dim = " + str(self.etype_dim) + '\n')
        file_object.write("syntax_dim = " + str(self.syntax_dim) + '\n')
        file_object.write("max_sent_size = " + str(self.max_sent_size) + '\n')
        file_object.write("max_edu_size = " + str(self.max_edu_size) + '\n')
        file_object.write("max_state_size = " + str(self.max_state_size) + '\n')
        file_object.write("hidden_size = " + str(self.hidden_size) + '\n')
        file_object.write("hidden_size_tagger = " + str(self.hidden_size_tagger) + '\n')
        
        file_object.write("freeze = " + str(self.freeze) + '\n')
        file_object.write("drop_prob = " + str(self.drop_prob) + '\n')
        file_object.write("num_layers = " + str(self.num_layers) + '\n')

        file_object.write("batch_size = " + str(self.batch_size) + '\n')
        file_object.write("opt = " + str(self.opt) + '\n')
        file_object.write("lr = " + str(self.lr) + '\n')
        file_object.write("ada_eps = " + str(self.ada_eps) + '\n')
        file_object.write("momentum = " + str(self.momentum) + '\n')
        file_object.write("beta1 = " + str(self.beta1) + '\n')
        file_object.write("beta2 = " + str(self.beta2) + '\n')
        file_object.write("gamma = " + str(self.gamma) + '\n')
        file_object.write("start_decay = " + str(self.start_decay) + '\n')
        file_object.write("grad_accum = " + str(self.grad_accum) + '\n')

        file_object.write("clip = " + str(self.clip) + '\n')
        
        file_object.write("loss_nuc_rel = " + str(self.loss_nuc_rel) + '\n')
        file_object.write("loss_seg = " + str(self.loss_seg) + '\n')
        file_object.write("activate_nuc_rel_loss = " + str(self.activate_nuc_rel_loss) + '\n')
        file_object.write("activate_seg_loss = " + str(self.activate_seg_loss) + '\n')

        file_object.write("decay = " + str(self.decay) + '\n')
        file_object.write("oracle_prob = " + str(self.oracle_prob) + '\n')
        file_object.write("start_dynamic_oracle = " + str(self.start_dynamic_oracle) + '\n')
        file_object.write("early_stopping = " + str(self.early_stopping) + '\n')
        
        file_object.write("beam_search = " + str(self.beam_search) + '\n')
        file_object.write("depth_alpha = " + str(self.depth_alpha) + '\n')
        file_object.write("elem_alpha = " + str(self.elem_alpha) + '\n')
        file_object.write("seed = " + str(self.seed) + '\n')

    def load_config(self, path):
        f = open(path, 'r')
        self.read(f)
        f.close()

    def read(self, file_object):
        self.architecture = file_object.readline().strip().split(' = ')[-1]
        self.use_gpu = file_object.readline().strip().split(' = ')[-1] == 'True'
        self.use_dynamic_oracle = file_object.readline().strip().split(' = ')[-1] == 'True'
        self.flag_oracle = file_object.readline().strip().split(' = ')[-1] == 'True'
        self.word_embedding = file_object.readline().strip().split(' = ')[-1] 
        self.word_embedding_file = file_object.readline().strip().split(' = ')[-1] 
    
        self.train_path = file_object.readline().strip().split(' = ')[-1] 
        self.test_path = file_object.readline().strip().split(' = ')[-1] 
        self.dev_path = file_object.readline().strip().split(' = ')[-1] 
        self.train_syn_feat_path = file_object.readline().strip().split(' = ')[-1] 
        self.test_syn_feat_path = file_object.readline().strip().split(' = ')[-1] 
        self.dev_syn_feat_path = file_object.readline().strip().split(' = ')[-1] 
        self.model_path = file_object.readline().strip().split(' = ')[-1] 
        self.model_name = file_object.readline().strip().split(' = ')[-1] 
        self.alphabet_path = file_object.readline().strip().split(' = ')[-1] 

        self.max_iter = int(file_object.readline().strip().split(' = ')[-1])
        self.word_dim = int(file_object.readline().strip().split(' = ')[-1])
        self.tag_dim = int(file_object.readline().strip().split(' = ')[-1])
        self.etype_dim = int(file_object.readline().strip().split(' = ')[-1])
        self.syntax_dim = int(file_object.readline().strip().split(' = ')[-1])
        self.max_sent_size = int(file_object.readline().strip().split(' = ')[-1])
        self.max_edu_size = int(file_object.readline().strip().split(' = ')[-1])
        self.max_state_size = int(file_object.readline().strip().split(' = ')[-1])
        self.hidden_size = int(file_object.readline().strip().split(' = ')[-1])
        self.hidden_size_tagger = int(file_object.readline().strip().split(' = ')[-1])
        
        self.freeze = file_object.readline().strip().split(' = ')[-1] == 'True'
        self.drop_prob = float(file_object.readline().strip().split(' = ')[-1])
        self.num_layers = int(file_object.readline().strip().split(' = ')[-1])

        self.batch_size = int(file_object.readline().strip().split(' = ')[-1])
        self.opt = file_object.readline().strip().split(' = ')[-1] 
        self.lr = float(file_object.readline().strip().split(' = ')[-1])
        self.ada_eps = float(file_object.readline().strip().split(' = ')[-1])
        self.momentum = float(file_object.readline().strip().split(' = ')[-1])
        self.beta1 = float(file_object.readline().strip().split(' = ')[-1])
        self.beta2 = float(file_object.readline().strip().split(' = ')[-1])
        self.betas = (self.beta1, self.beta2)
        self.gamma = float(file_object.readline().strip().split(' = ')[-1])
        self.start_decay = int(file_object.readline().strip().split(' = ')[-1])
        self.grad_accum = int(file_object.readline().strip().split(' = ')[-1])

        self.clip = float(file_object.readline().strip().split(' = ')[-1])
        
        self.loss_nuc_rel = float(file_object.readline().strip().split(' = ')[-1])
        self.loss_seg = float(file_object.readline().strip().split(' = ')[-1])
        self.activate_nuc_rel_loss = int(file_object.readline().strip().split(' = ')[-1])
        self.activate_seg_loss = int(file_object.readline().strip().split(' = ')[-1])

        self.decay = int(file_object.readline().strip().split(' = ')[-1])
        self.oracle_prob = float(file_object.readline().strip().split(' = ')[-1])
        self.start_dynamic_oracle = int(file_object.readline().strip().split(' = ')[-1])
        self.early_stopping = int(file_object.readline().strip().split(' = ')[-1])
        
        self.beam_search = int(file_object.readline().strip().split(' = ')[-1])
        self.depth_alpha = float(file_object.readline().strip().split(' = ')[-1])
        self.elem_alpha = float(file_object.readline().strip().split(' = ')[-1])
        self.seed = int(file_object.readline().strip().split(' = ')[-1])


class TopDownConfig(Config):
    pass


class BottomUpConfig(Config):
    def __init__(self, args):
        super().__init__(args)
        if args is None:
            return
        self.subtree_order_for_training = args.subtree_order
        self.target_merges = args.target_merges
        if args.merge_inference.split('-')[0] == 'threshold':
            self.merge_selection_for_inference, threshold = args.merge_inference.split('-')
            self.merge_selection_threshold = int(threshold) / 100
        else:
            self.merge_selection_for_inference = args.merge_inference
            self.merge_selection_threshold = 1

    def dump(self, file_object):
        super().dump(file_object)
        file_object.write("subtree_order = " + self.subtree_order_for_training + '\n')
        file_object.write("target_merges = " + self.target_merges + '\n')
        file_object.write("merge_selection_for_inference = " + self.merge_selection_for_inference + '\n')
        file_object.write("merge_selection_threshold = " + str(self.merge_selection_threshold) + '\n')

    def read(self, file_object):
        super().read(file_object)
        self.subtree_order_for_training = file_object.readline().strip().split(' = ')[-1]
        self.target_merges = file_object.readline().strip().split(' = ')[-1]
        self.merge_selection_for_inference = file_object.readline().strip().split(' = ')[-1]
        self.merge_selection_threshold = float(file_object.readline().strip().split(' = ')[-1])


def get_config(args):
    if args.architecture == 'top-down':
        return TopDownConfig(args)
    elif args.architecture == 'bottom-up':
        return BottomUpConfig(args)
    else:
        raise NotImplementedError(f'Architecture {args.architecture} is not implemented')

def load_config(path):
    f = open(path, 'r')
    architecture = f.readline().strip().split(' = ')[-1]
    f.close()
    if architecture == 'top-down':
        config = TopDownConfig(None)
    elif architecture == 'bottom-up':
        config = BottomUpConfig(None)
    else:
        raise NotImplementedError(f'Architecture {architecture} is not implemented')
    config.load_config(path)
    return config
