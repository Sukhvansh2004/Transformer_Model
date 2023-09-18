import torch
import torch.nn as nn
from transformers_layer import *

device = 'cpu'
class swin(nn.Module):
    """Implement the SWIN Transformer layer. The layer architexture consists of a Layer Norm, MSA, Layer Norm, MLP with residual connection"""
    def __init__(self, dim, heads, window_size, device=device):
        """Inputs:
        - dim : Dimesion of the input 
        - heads : No. of heads to be divided in for MSA
        - window_size : window dimension of the MSA"""
        super().__init__()
        self.layernorm1 = nn.LayerNorm(dim).to(device=device)
        self.MSA = MultiHeadAttention(dim=dim, num_heads=heads, window_size=window_size).to(device=device)
        self.layernorm2 = nn.LayerNorm(dim).to(device=device)
        self.MLP = multi_layer_perceptron(dim[0] * dim[1] * dim[2], device=device).to(device=device)
    
    def forward(self, features):
        """Forward Implementation of the SWIN layer on the feature vectors"""
        x = self.layernorm1(features)
        attention_features = self.MSA(x) + x
        x = self.layernorm2(attention_features)
        shape = x.shape
        return self.MLP(x.reshape(shape[0], shape[1] * shape[2] * shape[3])).reshape(shape) + x
        

class multi_layer_perceptron(nn.Module):
    def __init__(self,embed_dim, device=device):
        """Fully Connected 2 layer neural network with GELU non linearity in between
        
        Inputs:
        - embed_dim: Dimension of the inputs"""
        
        super().__init__()
        self.layer1 = nn.Linear(embed_dim, embed_dim).to(device=device)
        self.non_linearity = nn.GELU().to(device=device)
        self.layer2 = nn.Linear(embed_dim, embed_dim).to(device=device)
        
    def forward(self,x):
        """Forward Layer Implementation of the Multi Layer Perceptron"""
        
        x = self.layer1(x)
        x = self.non_linearity(x)
        x = self.layer2(x)
        return x
