import torch
import torch.nn as nn
from transformers_layer import *

class swin(nn.Module):
    """Implement the SWIN Transformer layer. The layer architexture consists of a Layer Norm, MSA, Layer Norm, MLP with residual connection"""
    def __init__(self, dim, heads):
        """Inputs:
        - dim : Dimesion of the input and output
        - heads : No. of heads to be divided in for MSA"""
        super().__init__()
        self.layernorm1 = nn.LayerNorm(dim)
        self.MSA = MultiHeadAttention(dim=dim, num_heads=heads)
        self.layernorm2 = nn.LayerNorm(dim)
        self.MLP = multi_layer_perceptron(dim)
    
    def forward(self, features):
        """Forward Implementation of the SWIN layer on the feature vectors"""
        x = self.layernorm1(features)
        attention_features = self.MSA(query = x, key = x, value = x) + x
        x = self.layernorm2(attention_features)
        return self.MLP(x) + x
        

class multi_layer_perceptron(nn.Module):
    def __init__(self,embed_dim,):
        super().__init__()
        self.layer1 = nn.Linear(embed_dim, embed_dim)
        self.non_linearity = nn.GELU()
        self.layer2 = nn.Linear(embed_dim, embed_dim)
        
    def forward(self,x):
        x = self.layer1(x)
        x = self.non_linearity(x)
        x = self.layer2
        return x
