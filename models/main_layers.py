import torch
import torch.nn as nn
import torch.nn.functional as F
from models.activations import Swish
from models.utils import SeparableConv1d, SeparableConv2d, MixerBlock
import math 

## convolutional mixer block    
class ConvMixerBlock(nn.Module):
    """ Performs convolution with mlp mixer. Processing steps ::
        1) freq depthwise separable convolution
        2) time depthwise separable convolution
        3) mlp mixer
        4) skip connection
    
    """
    def __init__(self, temporal_length, num_temporal_channels=64, 
                 temporal_kernel_size=3, temporal_padding=1,
                 freq_domain_kernel_size=5, freq_domain_padding=2,
                 num_freq_filters=64, 
                 dropout=0.,
                 bias=False):
        super(ConvMixerBlock, self).__init__()

        ## frequency domain encoding
        self.frequency_domain_encoding = nn.Sequential(
            nn.Conv2d(1, num_freq_filters, kernel_size=3, stride=1, padding=1, bias=bias),
            Swish(),
            SeparableConv2d(num_freq_filters, num_freq_filters, 
                            kernel_size=(freq_domain_kernel_size, 1),
                            stride=1, padding=(freq_domain_padding, 0), bias=bias),
            Swish(), 
            nn.Conv2d(num_freq_filters, 1, kernel_size=1, stride=1, padding=0, bias=bias), 
            nn.BatchNorm2d(1),
            Swish(),)

        ## temporal domain encoding
        self.temporal_domain_encoding = nn.Sequential(
            SeparableConv1d(num_temporal_channels, num_temporal_channels, 
                            kernel_size=temporal_kernel_size,
                            stride=1, padding=temporal_padding, bias=bias),
            nn.BatchNorm1d(num_temporal_channels),
            Swish(),)
        
        self.dropout = nn.Dropout(p=dropout)
        
        ## mixer
        self.mixer = nn.Sequential(
            MixerBlock(time_dim=temporal_length, freq_dim=num_temporal_channels, dropout=0.),
            Swish(),)
        
    def forward(self, x):
        skipInput = x
        skipInput2 = x = self.dropout(self.frequency_domain_encoding(x.unsqueeze(1)).squeeze(1))
        x = self.dropout(self.temporal_domain_encoding(x))
        x = self.mixer(x)
        
        return skipInput + skipInput2 + x
        

## convolutional mixer block 
class PreConvBlock(nn.Module):

    def __init__(self, time_length, time_channels=64, 
                 kernel_size=3, padding=1, 
                 dropout=0., bias=False):
        
        super(PreConvBlock, self).__init__()
        
        ## temporal domain encoding
        self.temporal_domain_encoding = nn.Sequential(
            SeparableConv1d(time_channels, time_channels, kernel_size=kernel_size,
                            stride=1, padding=padding, bias=bias),
            nn.BatchNorm1d(time_channels),
            Swish(),
            SeparableConv1d(time_channels, time_channels, kernel_size=1,
                            stride=1, padding=0, bias=bias),
            nn.BatchNorm1d(time_channels),
            Swish(),)
        
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x):
        skipInput = x
        x = self.dropout(self.temporal_domain_encoding(x))
        
        return skipInput + x