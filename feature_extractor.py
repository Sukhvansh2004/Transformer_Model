import torch
import torch.nn as nn
import RSTB

class shallow_feature(nn.Module):
    """Class Containing the shallow feature extractor"""
    def __init__(self, input_dimension, output_dimension):
        """
        Inputs:
        -input_dimension: Tuple containing the input features C of the image
        -output_dimension:Tuple containing the output features C' of the image"""
        super().__init__()
        self.Convolution = nn.Conv2d(in_channels = input_dimension, out_channels = output_dimension, kernel_size = 3, padding = "same")
        
    def forward(self, image):
        """Forward function of the shallow feature extractor"""
        return self.Convolution(image)

class deep_features(nn.Module):
    "Class containing the deep feature extractor"
    def __init__(self, n, heads, dim):
        """Inputs:
        -n : An array having the no. of elements in each RSTB
        -heads: An array of having an array as each element for heads
        -dim: dimension of the input features"""
        super().__init__()
        self.layers = {}
        self.n  = len(n)
        for i in range(1, len(n)+1):
            self.layers[i] = RSTB.RSTB(dim[0], dim[1], dim[2], n[i-1], heads[i-1])
        self.Convolution = nn.Conv2d(dim[0], dim[0], 3, padding = 'same')
        
    def forward(self, features):
        x = features
        for i in range(1, self.n + 1):
            x = self.layers[i](x)
        return self.Convolution(x) + features            
        
class feature(nn.Module):
    "Feature extractor model of the Transformer"
    def __init__(self, input_dimension, output_dimension, n, heads):
        """Check the initialisation of the deep_feature and shallow_feature extractor functions"""
        super().__init__()
        self.shallow = shallow_feature(input_dimension, output_dimension)
        self.deep = deep_features(n, heads, output_dimension)
        
    def forward(self, img):
        return self.deep(self.shallow(img))
    
