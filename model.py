import torch
import torch.nn as nn
from feature_extractor import *
import numpy as np

class SWINIR(nn.module):
    def __init__(self, dim, RSTB_nos = 6, STL_nos = 6, window_size = (8, 8), channel_nos = 180, attn_head = 6):
    
        """Implementation of the SWINIR Model:
        
        Inputs:
        - dim: dimension of the input image
        - RSTB_nos: No. of RSTB in the deep feature extractor
        - STL_nos: No. of STL in a RSTB 
        - window_size: Size of a window in the MSA
        - channel_nos: No. of feature channels
        - attn_head: No. of attention heads for the MSA"""
        
        super().__init__()
        self.extractor = feature(input_dimension = dim, output_dimension = (channel_nos, dim[1], dim[2]), n = np.ones(RSTB_nos) * STL_nos, heads = np.ones((RSTB_nos, STL_nos)) * attn_head, window_size = window_size)
        self.HQ_Reconstruction = nn.Sequential(
            nn.ConvTranspose2d(channel_nos, channel_nos//4, 4, 2, padding=1),
            nn.Conv2d(channel_nos//4, channel_nos//4, 3, padding="same"),
            nn.ConvTranspose2d(channel_nos//4, channel_nos//16, 4, 2, padding=1),
            nn.Conv2d(channel_nos//16, channel_nos//16, 3, padding="same"),
            nn.ConvTranspose2d(channel_nos//16, dim[0], 3),
            nn.Conv2d(dim[0], dim[0], 3, padding="same")
        )
        
    def forward(self, img):
        return self.HQ_Reconstruction(self.extractor(img))