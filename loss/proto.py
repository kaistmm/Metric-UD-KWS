'''
Copyright (c) 2020-present NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
'''

#! /usr/bin/python
# -*- encoding: utf-8 -*-
## Re-implementation of prototypical networks (https://arxiv.org/abs/1703.05175).
## Numerically checked against https://github.com/cyvius96/prototypical-network-pytorch

import torch
import torch.nn as nn
import torch.nn.functional as F
import time, pdb, numpy
from accuracy import accuracy

class LossFunction(nn.Module):

    def __init__(self, fine_tunning, **kwargs):
        super(LossFunction, self).__init__()

        self.test_normalize = False
        self.fine_tunning = fine_tunning

        self.nOut = kwargs['nOut']
        self.nClasses = kwargs['nClasses']

        if self.fine_tunning == True:
            self.fc = nn.Linear(self.nOut, self.nClasses)

        self.criterion  = torch.nn.CrossEntropyLoss()

        print('Initialised Prototypical Loss')

    def forward(self, x, label=None):

        assert x.size()[1] >= 2

        if self.fine_tunning == True:
            x = self.fc(x)
        
        out_anchor      = torch.mean(x[:,1:,:],1)
        out_positive    = x[:,0,:]
        stepsize        = out_anchor.size()[0]

        output      = -1 * (F.pairwise_distance(out_positive.unsqueeze(-1),out_anchor.unsqueeze(-1).transpose(0,2))**2)
        label       = torch.from_numpy(numpy.asarray(range(0,stepsize))).cuda()
        nloss       = self.criterion(output, label)
        prec1, _    = accuracy(output.detach().cpu(), label.detach().cpu(), topk=(1, 2))

        return nloss, prec1