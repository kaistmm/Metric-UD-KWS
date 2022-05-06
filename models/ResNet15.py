#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from utils import PreEmphasis
import pdb

class ResNet15(nn.Module):
    def __init__(self, nOut, **kwargs):
        super(ResNet15, self).__init__()

        n_labels = nOut
        n_maps = 45 #차원 수
        dilation = True # config["use_dilation"]

        self.conv0 = nn.Conv2d(1, n_maps, (3,3), padding=(1,1), bias=False)
        self.n_layers = n_layers = 13 # 2x6(res) + 1(conv)

        if dilation:
            self.convs = [nn.Conv2d(n_maps, n_maps, (3, 3), padding=int(2**(i // 3)), dilation=int(2**(i // 3)),
                bias=False) for i in range(n_layers)]
        else:
            self.convs = [nn.Conv2d(n_maps, n_maps, (3, 3), padding=1, dilation=1,
                bias=False) for _ in range(n_layers)]

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
    model = ResNet15(nOut, **kwargs)
    return model