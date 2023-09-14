import torch
import torch.nn as nn
from torch.nn import functional as F
import math

"""
This file defines layer types that are commonly used for transformers.
"""

class PositionalEncoding(nn.Module):
    """
    Encodes information about the positions of the tokens in the sequence. In
    this case, the layer has no learnable parameters, since it is a simple
    function of sines and cosines.
    """
    def __init__(self, embed_dim, dropout=0.1, max_len=5000):
        """
        Construct the PositionalEncoding layer.

        Inputs:
         - embed_dim: the size of the embed dimension
         - dropout: the dropout value
         - max_len: the maximum possible length of the incoming sequence
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        assert embed_dim % 2 == 0
        pe = torch.zeros(1, max_len, embed_dim)
       
        
        row = torch.arange(max_len).unsqueeze(1)
        col_sin = torch.arange(0, embed_dim , 2).unsqueeze(0)
        col_cos = torch.arange(1, embed_dim , 2).unsqueeze(0)
        pe[ 0, :, col_sin[0]] = torch.sin((row) * 10000**(-col_sin/embed_dim))
        pe[ 0, :, col_cos[0]] = torch.cos((row) * 10000**(-(col_cos-1)/embed_dim))
            
        pass
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Element-wise add positional embeddings to the input sequence.

        Inputs:
         - x: the sequence fed to the positional encoder model, of shape
              (N, S, D), where N is the batch size, S is the sequence length and
              D is embed dim
        Returns:
         - output: the input sequence + positional encodings, of shape (N, S, D)
        """
        N, S, D = x.shape
        output = torch.empty((N, S, D))
        
        output = self.dropout(x + self.pe[:,:S,:D]) 
        pass
    
        return output


class MultiHeadAttention(nn.Module):
    """
    A model layer which implements a simplified version of masked attention, as
    introduced by "Attention Is All You Need" (https://arxiv.org/abs/1706.03762).

    Usage:
      attn = MultiHeadAttention(embed_dim, num_heads=2)

      # self-attention
      data = torch.randn(batch_size, sequence_length, embed_dim)
      self_attn_output = attn(query=data, key=data, value=data)

      # attention using two inputs
      other_data = torch.randn(batch_size, sequence_length, embed_dim)
      attn_output = attn(query=data, key=other_data, value=other_data)
    """

    def __init__(self, dim, num_heads, dropout=0.1):
        """
        Construct a new MultiHeadAttention layer.

        Inputs:
         - dim: Dimension of the input 
         - num_heads: Number of attention heads
         - dropout: Dropout probability
        """
        super().__init__()
        assert dim % num_heads == 0

        self.key = nn.Linear(dim, dim)
        self.query = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        
        self.attn_drop = nn.Dropout(dropout)

        self.n_head = num_heads
        self.dim = dim
        self.head_dim = self.dim // self.n_head

    def forward(self, query, key, value, attn_mask=None):
        """
        Calculate the masked attention output for the provided data, computing
        all attention heads in parallel.

        In the shape definitions below, N is the batch size, S is the source
        sequence length, T is the target sequence length, and E is the embedding
        dimension.

        Inputs:
        - query: Input data to be used as the query, of shape (N, S, E)
        - key: Input data to be used as the key, of shape (N, T, E)
        - value: Input data to be used as the value, of shape (N, T, E)
        - attn_mask: Array of shape (S, T) where mask[i,j] == 0 indicates token
          i in the source should not influence token j in the target.

        Returns:
        - output: Tensor of shape (N, S, E) giving the weighted combination of
          data in value according to the attention weights calculated using key
          and query.
        """
        N, S, E = query.shape
        N, T, E = value.shape
        
        output = torch.empty((N, S, E))
        
        key = self.key(key)
        query = self.query(query)
        value = self.value(value)
     
        Q = query.view(N, S, self.n_head, self.head_dim).moveaxis(1,2)
        K = key.view(N, T, self.n_head, self.head_dim).moveaxis(1,2)
        V = value.view(N, T, self.n_head, self.head_dim).moveaxis(1,2)
        
        scores = torch.matmul(Q, K.transpose(2,3))/(self.head_dim**(1/2))
        if attn_mask is not None:
          scores = scores.masked_fill(attn_mask==0, float(-torch.inf))
          pass
        probs = F.softmax(scores,dim=-1)
        output = self.proj(torch.matmul(self.attn_drop(probs), V).moveaxis(1,2).reshape(N, S, E))


        return output
      

