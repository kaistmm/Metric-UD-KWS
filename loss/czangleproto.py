#! /usr/bin/python
# -*- encoding: utf-8 -*-
## Implementation of cross-zero NCE on angle prototypical loss (https://arxiv.org/pdf/2202.13083.pdf)
## Jihwan Park (jihwan.park@hyundai.com), June 8, 2022.

import torch
import torch.nn as nn
import torch.nn.functional as F
import time, pdb, numpy
from accuracy import accuracy

class LossFunction(nn.Module):

    def __init__(self, init_w=10.0, init_b=-5.0, **kwargs):
        super(LossFunction, self).__init__()

        self.test_normalize = True
        
        self.w = nn.Parameter(torch.tensor(init_w))
        self.b = nn.Parameter(torch.tensor(init_b))
        self.criterion  = torch.nn.CrossEntropyLoss()

        print('Initialised CrossZeroAngleProto')

    def forward(self, x, label=None): #x.shape = torch.Size([20, 2, 12])
        assert x.size()[1] >= 2

        out_anchor      = torch.mean(x[:,1:,:],1) # out_anchor = torch.Size([20, 20])  #왜 mean을 쓴거지?
        out_positive    = x[:,0,:] #out_postive = torch.Size([20, 20])
        stepsize        = out_anchor.size()[0] #20

        cos_sim_matrix  = F.cosine_similarity(out_positive.unsqueeze(-1),out_anchor.unsqueeze(-1).transpose(0,2)) 
        #out_positive.unsqueeze(-1).shape = torch.Size([20, 10, 1])
        #out_anchor.unsqueeze(-1).transpose(0,2).shape = torch.Size([1, 20, 20])
        #cos_sim = torch.Size([20, 20])
        torch.clamp(self.w, 1e-6)
                
        cos_sim_matrix = cos_sim_matrix * self.w + self.b

        label = torch.from_numpy(numpy.asarray(range(0,stepsize))).cuda() #label.shape = torch.Size([20])
        temp = 0
        for i in range(0, stepsize):
            temp += torch.logsumexp(torch.concat([cos_sim_matrix[:,i][:i], cos_sim_matrix[:,i][i+1:]]),0) - cos_sim_matrix[i,i]
        nloss = temp / stepsize

        prec1, _    = accuracy(cos_sim_matrix.detach().cpu(), label.detach().cpu(), topk=(1, 2))
        # pdb.set_trace()

        return nloss, prec1