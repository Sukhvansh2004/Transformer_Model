import torch
import torch.nn as nn
import feature_extractor
import numpy as np

class SWINIR(nn.module):
    def __init__(self, dim, RSTB_nos = 6, STL_nos = 6, window_size = (8, 8), channel_nos = 180, attn_head = 6):
        super().__init__()
        self.extractor = feature_extractor.feature(input_dimension = dim, output_dimension = (channel_nos, dim[1], dim[2]), n = np.ones(RSTB_nos) * STL_nos, heads = np.ones((RSTB_nos, STL_nos)) * attn_head, window_size = window_size)
        
    def forward(self, img):
        return self.extractor(img)