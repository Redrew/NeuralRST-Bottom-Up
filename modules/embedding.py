import math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import torch.nn.functional as F
def assign_tensor(tensor, val):
    """
    copy val to tensor
    Args:
        tensor: an n-dimensional torch.Tensor or autograd.Variable
        val: an n-dimensional torch.Tensor to fill the tensor with

    Returns:
    """
    if isinstance(tensor, Variable):
        assign_tensor(tensor.data, val)
        return tensor
    return tensor.copy_(val)


def chunkify(edu_lengths, max_chunk_length):
    """
    split span into chunks along edu boundaries
    """
    chunk_boundaries = []
    start, end = 0, 0
    for edu_length in edu_lengths:
        if end - start + edu_length > max_chunk_length:
            chunk_boundaries.append((start, end))
            start = end
        end += int(edu_length)
    if end - start > 0:
        chunk_boundaries.append((start, end))

    return chunk_boundaries

def chunk_end(edu_lengths, max_chunk_length):
    end = int(sum(edu_lengths))
    start = end
    for edu_length in reversed(edu_lengths):
        if end - start + edu_length > max_chunk_length:
            break
        start -= edu_length
    return (start, end)


class Embedding(nn.Module):
    r"""A simple lookup table that stores embeddings of a fixed dictionary and size.
    This module is often used to store word embeddings and retrieve them using indices.
    The input to the module is a list of indices, and the output is the corresponding
    word embeddings.
    Args:
        num_embeddings (int): size of the dictionary of embeddings
        embedding_dim (int): the size of each embedding vector
        init_embedding (Tensor or Variable): If given, the embedding will be initialized with the given tensor.
        freeze (boolean, optional): If ``True``, the tensor does not get updated in the learning process.
        padding_idx (int, optional): If given, pads the output with zeros whenever it encounters the index.
        max_norm (float, optional): If given, will renormalize the embeddings to always have a norm lesser than this
        norm_type (float, optional): The p of the p-norm to compute for the max_norm option
        scale_grad_by_freq (boolean, optional): if given, this will scale gradients by the frequency of
                                                the words in the mini-batch.
        sparse (boolean, optional): if True, gradient w.r.t. weight matrix will be a sparse tensor. See Notes for
                                    more details regarding sparse gradients.
    Attributes:
        weight (Tensor): the learnable weights of the module of shape (num_embeddings, embedding_dim)
    Shape:
        - Input: LongTensor `(N1, N2, ...,Nm, W)`, N = mini-batch, W = number of indices to extract per mini-batch
        - Output: `(N1, N2, ..., Nm, W, embedding_dim)`
    Notes:
        Keep in mind that only a limited number of optimizers support
        sparse gradients: currently it's `optim.SGD` (`cuda` and `cpu`),
        and `optim.Adagrad` (`cpu`)
    """

    def __init__(self, num_embeddings, embedding_dim, init_embedding=None, freeze=False, padding_idx=None,
                 max_norm=None, norm_type=2, scale_grad_by_freq=False, sparse=False):
        super(Embedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.weight = Parameter(torch.Tensor(num_embeddings, embedding_dim))
        self.frozen = freeze
        self.sparse = sparse

        self.reset_parameters(init_embedding)

    def reset_parameters(self, init_embedding):
        if init_embedding is None:
            scale = np.sqrt(3.0 / self.embedding_dim)
            self.weight.data.uniform_(-scale, scale)
        else:
            self.weight.data = init_embedding.clone()
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)

        if self.frozen:
            if init_embedding is None:
                raise Warning('Freeze embeddings which are randomly initialized.')
            self.weight.requires_grad = False

    def freeze(self):
        self.weight.requires_grad = False
        self.frozen = True

    def forward(self, input):
        padding_idx = self.padding_idx
        if padding_idx is None:
            padding_idx = -1

        input_size = input.size()
        if input.dim() > 2:
            num_inputs = int(np.prod(input_size[:-1]))
            input = input.view(num_inputs, input_size[-1])

        output_size = input_size + (self.embedding_dim,)
        return F.embedding(
            input, self.weight,
            padding_idx, self.max_norm, self.norm_type,
            self.scale_grad_by_freq, self.sparse).view(output_size)

    def __repr__(self):
        s = '{name}({num_embeddings}, {embedding_dim}'
        if self.padding_idx is not None:
            s += ', padding_idx={padding_idx}'
        if self.max_norm is not None:
            s += ', max_norm={max_norm}'
        if self.norm_type != 2:
            s += ', norm_type={norm_type}'
        if self.scale_grad_by_freq is not False:
            s += ', scale_grad_by_freq={scale_grad_by_freq}'
        if self.sparse is not False:
            s += ', sparse=True'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)

class ContextualEmbedding(nn.Module):
    def __init__(self, model, input_limit=np.inf, min_chunk_size=None, freeze=True, prefix=None, postfix=None):
        super(ContextualEmbedding, self).__init__()
        self.model = model.eval()
        self.embedding_dim = self.model.config.hidden_size
        prefix = [] if prefix is None else [prefix]
        postfix = [] if postfix is None else [postfix]
        self.prefix = nn.Parameter(torch.LongTensor(prefix), requires_grad=False)
        self.postfix = nn.Parameter(torch.LongTensor(postfix), requires_grad=False)
        self.input_limit = input_limit - self.prefix.shape[0] - self.postfix.shape[0]
        self.min_chunk_size = min_chunk_size if min_chunk_size else 0.5 * self.input_limit

        self.frozen = freeze
        self.model.requires_grad_(not freeze)

    def freeze(self):
        self.frozen = True
        self.model.requires_grad_(False)
    
    def forward(self, input, mask):
        embeddings = torch.zeros(*input.shape, self.embedding_dim).type(torch.FloatTensor)
        embeddings = embeddings.to(input.device)
        num_instances = input.shape[0]
        for i in range(num_instances):
            self.embed_instance(input[i], mask[i], embeddings[i])
        return embeddings
    
    def embed_instance(self, input, mask, output):
        mask = mask.bool()
        tokens = input[mask]
        embedded_chunks = []
        chunk_boundaries = chunkify(mask.sum(-1), self.input_limit)
        for chunk_i, (start, end) in enumerate(chunk_boundaries):
            last_chunk = chunk_i == len(chunk_boundaries) - 1
            if last_chunk and end - start < self.min_chunk_size: # the last chunk should be larger than min chunk size
                context_start, context_end = chunk_end(mask.sum(-1), self.input_limit)
                assert (end == context_end)
                embedded_chunk = self.embed_tokens(tokens[context_start:context_end])[start - context_start:]
            else:
                embedded_chunk = self.embed_tokens(tokens[start:end])
            embedded_chunks.append(embedded_chunk)
        embeddings = torch.cat(embedded_chunks)
        output[mask] = embeddings

    def embed_tokens(self, tokens):
        tokens = torch.cat([self.prefix, tokens, self.postfix])
        tokens = torch.unsqueeze(tokens, 0)
        model_out = self.model(input_ids=tokens)
        embeddings = model_out.last_hidden_state[0]
        prefix_end = self.prefix.shape[0]
        postfix_start = embeddings.shape[0] - self.postfix.shape[0]
        embeddings = embeddings[prefix_end:postfix_start]
        return embeddings

    def __repr__(self):
        pass