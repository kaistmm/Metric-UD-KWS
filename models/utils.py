import torch
import torch.nn as nn
import torch.nn.functional as F
from models.activations import Swish
import math 
            
## depthwise separable convolution (1D)
class SeparableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size, stride=1, padding=0,
                 dilation=1, bias=False, pointwise=True):
    
        super(SeparableConv1d, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=in_channels, out_channels=in_channels,
                                kernel_size=kernel_size, stride=stride, groups=in_channels, padding=padding,
                                dilation=dilation, bias=bias,)
        
        if pointwise:
            self.pointwise = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                                       kernel_size=1, stride=1, padding=0, bias=bias,)
        else:
            self.pointwise = nn.Identity()

    def forward(self, x):
        x = self.conv1d(x)
        x = self.pointwise(x)
        return x
    
    
## depthwise separable convolution (2D)    
class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size, stride=1, padding=0,
                 dilation=1, bias=False, pointwise=False):
        
        super(SeparableConv2d, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                                kernel_size=kernel_size, stride=stride, groups=in_channels, padding=padding,
                                dilation=dilation, bias=bias,)
        if pointwise:
            self.pointwise = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                       kernel_size=1, stride=1, padding=0, bias=bias,)
        else:
            self.pointwise = nn.Identity()
    
    def forward(self, x):
        x = self.conv2d(x)
        x = self.pointwise(x)
        return x

    
## MLP layer
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        """ Perform feed forward layer for mixer
        Parameter args:
            dim: in_channel dimension
            hidden_dim: intermediate dimension during FFN
        """
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout))
        
    def forward(self, x):
        return self.net(x)
    
    
## mixer block  
class MixerBlock(nn.Module):
    def __init__(self, time_dim, freq_dim, dropout=0.):
        super(MixerBlock, self).__init__()

        self.time_mix = nn.Sequential(
            nn.LayerNorm(time_dim),
            FeedForward(time_dim, time_dim // 4, dropout),)
        
        self.freq_mix = nn.Sequential(
            nn.LayerNorm(freq_dim), 
            FeedForward(freq_dim, freq_dim // 2, dropout),)

    def forward(self, x):
        x = x + self.time_mix(x)
        x = x + self.freq_mix(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x
    

