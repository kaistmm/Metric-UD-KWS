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

import torch
import torch.nn as nn
import torch.nn.functional as F
import time, pdb, numpy
from accuracy import accuracy

class LossFunction(nn.Module):

    def __init__(self, fine_tunning, init_w=10.0, init_b=-5.0, **kwargs):
        super(LossFunction, self).__init__()

        self.test_normalize = True
        self.fine_tunning = fine_tunning

        if self.fine_tunning == True:
            self.fc = nn.Linear(nOut, nClasses)
        
        self.w = nn.Parameter(torch.tensor(init_w))
        self.b = nn.Parameter(torch.tensor(init_b))
        self.criterion  = torch.nn.CrossEntropyLoss()

        print('Initialised AngleProto')

    def forward(self, x, label=None): #x.shape = torch.Size([20, 2, 12])
        assert x.size()[1] >= 2

        if self.fine_tunning = True:
            x = self.fc(x)

        out_anchor      = torch.mean(x[:,1:,:],1) # out_anchor = torch.Size([20, 20])  #왜 mean을 쓴거지?
        out_positive    = x[:,0,:] #out_postive = torch.Size([20, 20])
        stepsize        = out_anchor.size()[0] #20

        cos_sim_matrix  = F.cosine_similarity(out_positive.unsqueeze(-1),out_anchor.unsqueeze(-1).transpose(0,2)) 
        #out_positive.unsqueeze(-1).shape = torch.Size([20, 10, 1])
        #out_anchor.unsqueeze(-1).transpose(0,2).shape = torch.Size([1, 20, 20])
        #cos_sim = torch.Size([20, 20])
        torch.clamp(self.w, 1e-6)
        cos_sim_matrix = cos_sim_matrix * self.w + self.b
        
        label       = torch.from_numpy(numpy.asarray(range(0,stepsize))).cuda() #label.shape = torch.Size([20])
        nloss       = self.criterion(cos_sim_matrix, label)
        prec1, _    = accuracy(cos_sim_matrix.detach().cpu(), label.detach().cpu(), topk=(1, 2))
        # pdb.set_trace()

        return nloss, prec1