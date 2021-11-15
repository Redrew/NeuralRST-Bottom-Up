# NeuralRST-BottomUp

## About the code
The codebase is based on [Koto et al. 2021](https://github.com/fajri91/NeuralRST-TopDown), we translate their top-down architecture for bottom-up parsing.
In turn, the encoder is designed based on [Yu et al., 2018](https://github.com/yunan4nlp/NNDisParser) where we use three main embeddings: 
1) Word embedding, initialized by [glove.6B.200d.txt.gz](https://nlp.stanford.edu/projects/glove/).
2) POS Tags embedding, initialized randomly.
3) Syntax Embedding from [BiAffine Dependency Parser](https://arxiv.org/abs/1611.01734). Please refer to [RSTExtractor](https://github.com/fajri91/RSTExtractor) to see how we extract it

## Dependencies 
1. Python 3.6
2. Run `pip install -r requirements.txt`

## Data and Resource

We use [English RST Tree Bank](https://catalog.ldc.upenn.edu/LDC2002T07). Please make sure you have a right 
to access this data. Our code uses the input of the binarized discourse tree, provided by Yu et al., 2018. 

In this repository, we do not provide you with the raw RST Tree Bank, but the binarized version split 
in train/dev/test based on [Yu et al., 2018](https://github.com/yunan4nlp/NNDisParser). We also provide
the extracted syntax feature for each data split. Please download them [here](https://drive.google.com/file/d/1mSS6Nj8vkiU9Q6q8r-I7p2fh44NOdFsZ/view).

## Running the code

In the experiment we use 1 GPU V100 (32GB).

For training the LSTM with static oracle (normal training)
```
CUDA_VISIBLE_DEVICES=0 python train_rst_parser.py bottom-up \
        --experiment=exp_static \
        --word_embedding_file=[path_to_glove] \
        --train=[path_to_train_data] --test=[path_to_test_data] --dev=[path_to_dev_data] \
        --train_syn_feat=[path_to_syntax_feature_of_train] \
        --test_syn_feat=[path_to_syntax_feature_of_test] \
        --dev_syn_feat=[path_to_syntax_feature_of_dev] \
        --subtree_order=random \
        --target_merges=all \
        --merge_inference=max \
        --word_embedding=glove \
        --max_sent_size=100 --hidden_size=256 --hidden_size_tagger=128 --batch_size=4 \
        --grad_accum=2 --lr=0.001 --ada_eps=1e-6 --gamma=1e-6 \
        --loss_seg=1.0 --loss_nuc_rel=1 \
        --depth_alpha=0 --elem_alpha=0 
```

For training the LSTM with the dynamic oracle:
```
CUDA_VISIBLE_DEVICES=0 python train_rst_parser.py bottom-up \
        --experiment=exp_dynamic \
        --word_embedding_file=[path_to_glove] \
        --train=[path_to_train_data] --test=[path_to_test_data] --dev=[path_to_dev_data] \
        --train_syn_feat=[path_to_syntax_feature_of_train] \
        --test_syn_feat=[path_to_syntax_feature_of_test] \
        --dev_syn_feat=[path_to_syntax_feature_of_dev] \
        --subtree_order=random \
        --target_merges=all \
        --merge_inference=max \
        --word_embedding=glove \
        --max_sent_size=100 --hidden_size=256 --hidden_size_tagger=128 --batch_size=4 \
        --grad_accum=2 --lr=0.001 --ada_eps=1e-6 --gamma=1e-6 \
        --loss_seg=1.0 --loss_nuc_rel=1 \
        --depth_alpha=0 --elem_alpha=0 \
        --use_dynamic_oracle=1 --start_dynamic_oracle=50 --oracle_prob=$1
```
