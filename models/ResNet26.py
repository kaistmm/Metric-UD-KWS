'''
* Original Code : https://github.com/castorini/honk/blob/master/utils/model.py
* modified by jjm

MIT License

Copyright (c) 2018 Castorini

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from utils import PreEmphasis
import pdb

class ResNet26(nn.Module):
    def __init__(self, nOut, **kwargs):
        super(ResNet26, self).__init__()

        n_labels = nOut
        n_maps = 45 #차원 수
        dilation = False # config["use_dilation"]

        self.conv0 = nn.Conv2d(1, n_maps, (3,3), padding=(1,1), bias=False)
        self.pool = nn.AvgPool2d((2,2))

        self.n_layers = 24 # 2x6(res) + 1(conv)

        if dilation:
            self.convs = [nn.Conv2d(n_maps, n_maps, (3, 3), padding=int(2**(i // 3)), dilation=int(2**(i // 3)),
                bias=False) for i in range(self.n_layers)]
        else:
            self.convs = [nn.Conv2d(n_maps, n_maps, (3, 3), padding=1, dilation=1,
                bias=False) for _ in range(self.n_layers)]

        for i, conv in enumerate(self.convs):
            self.add_module("bn{}".format(i + 1), nn.BatchNorm2d(n_maps, affine=False))
            self.add_module("conv{}".format(i + 1), conv)

        self.output = nn.Linear(n_maps, n_labels)

    def forward(self, x): #torch.Size([200, 201 , 40])

        x = x.unsqueeze(1) #배치 차원 추가 torch.Size([200, 1, 201, 40])
        
        for i in range(self.n_layers + 1):
            y = F.relu(getattr(self, "conv{}".format(i))(x)) #속성 가져오기
            if i == 0:
                if hasattr(self, "pool"): #속성이 존재하는지 확인 
                    y = self.pool(y)
                old_x = y
            if i > 0 and i % 2 == 0:
                x = y + old_x
                old_x = x
            else:
                x = y
            if i > 0:
                x = getattr(self, "bn{}".format(i))(x)
        x = x.view(x.size(0), x.size(1), -1) # shape: (batch, feats, o3)
        x = torch.mean(x, 2) #torch.Size([200, 45])

        return self.output(x)


def MainModel(nOut=256, **kwargs):
    # Number of filters
    model = ResNet26(nOut, **kwargs)
    return model