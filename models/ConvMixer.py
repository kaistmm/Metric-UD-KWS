#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Created on Oct 10, 2021, 17:23:06
# @author: dianwen ng
# @file  : ConvMixer.py


import torch
import torch.nn as nn
import torch.nn.functional as F
from models.activations import Swish
from models.utils import SeparableConv1d, SeparableConv2d
from models.main_layers import ConvMixerBlock, PreConvBlock
import math 
import pdb


class KWSConvMixer(nn.Module):
    def __init__(self, nOut, 
                 feat_dim=64,
                 dropout=0., **kwargs):
        
        """ KWS Convolutional Mixer Model
        input:: audio spectrogram, default input shape [BS, 98, 64]
        output:: prediction of command classes, default 12 classes
        """
        
        super(KWSConvMixer, self).__init__()
        self.nOut = nOut
        # self.temporal_dim, self.frequency_dim = input_size
        self.temporal_dim = 101     ## need to change
        self.frequency_dim = 40

        ## init conv (channel): output shape BS x feat_dim x T        
        self.conv1 = nn.Sequential(
            SeparableConv1d(self.frequency_dim, feat_dim, 
                            kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(feat_dim),
            Swish(),)
        
        self.preConvMixer = PreConvBlock(self.temporal_dim, feat_dim,
                                         kernel_size=7, padding=3,
                                         dropout=dropout)

        self.convMixer1 = ConvMixerBlock(self.temporal_dim, feat_dim,
                                         temporal_kernel_size=9, temporal_padding=4,
                                         freq_domain_kernel_size=5, freq_domain_padding=2,
                                         num_freq_filters=64, 
                                         dropout=dropout)
        
        self.convMixer2 = ConvMixerBlock(self.temporal_dim, feat_dim,
                                         temporal_kernel_size=11, temporal_padding=5,
                                         freq_domain_kernel_size=5, freq_domain_padding=2,
                                         num_freq_filters=32, 
                                         dropout=dropout)
            
        self.convMixer3 = ConvMixerBlock(self.temporal_dim, feat_dim,
                                         temporal_kernel_size=13, temporal_padding=6,
                                         freq_domain_kernel_size=7, freq_domain_padding=3,
                                         num_freq_filters=16, 
                                         dropout=dropout)
        
        self.convMixer4 = ConvMixerBlock(self.temporal_dim, feat_dim,
                                         temporal_kernel_size=15, temporal_padding=7,
                                         freq_domain_kernel_size=7, freq_domain_padding=3,
                                         num_freq_filters=8, 
                                         dropout=dropout)
        
        self.conv2 = nn.Sequential(
            SeparableConv1d(feat_dim, feat_dim*2, 
                            kernel_size=17, stride=1, padding=8, bias=False),
            nn.BatchNorm1d(feat_dim*2),
            Swish(),) 
        
        self.conv3 = nn.Sequential( 
            SeparableConv1d(feat_dim*2, feat_dim*2, 
                            kernel_size=19, stride=1, padding=18, dilation=2, bias=False),
            nn.BatchNorm1d(feat_dim*2),
            Swish(),) 
        
        self.conv4 = nn.Sequential( 
            SeparableConv1d(feat_dim*2, feat_dim*2, 
                            kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(feat_dim*2),
            Swish(),)

        self.pooling = torch.nn.AdaptiveMaxPool1d(1) 
        self.mlp_head = nn.Sequential(nn.Linear(feat_dim*2, self.nOut, bias=True))
        
        
    def forward(self, x):

        x = self.conv1(x.permute(0, 2, 1))  # x = (B, T, F)
        x = self.preConvMixer(x)
        x = self.convMixer1(x)
        x = self.convMixer2(x) 
        x = self.convMixer3(x) 
        x = self.convMixer4(x) 

        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        batch, in_channels, timesteps = x.size()
        x = self.pooling(x).view(batch, in_channels)
        
        return self.mlp_head(x)
    
    
def MainModel(nOut=256, **kwargs):
    model = KWSConvMixer(nOut, **kwargs)
    return model