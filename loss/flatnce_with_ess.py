#! /usr/bin/python
# -*- encoding: utf-8 -*-
## Implementation of flatNCE with effective sample size (ESS) (https://arxiv.org/pdf/2107.01152.pdf)
## Jihwan Park (jihwan.park@hyundai.com), July 8, 2022.

import torch
import torch.nn as nn
import torch.nn.functional as F
import time, pdb, numpy
from accuracy import accuracy

class LossFunction(nn.Module):

    def __init__(self, **kwargs):
        super(LossFunction, self).__init__()

        self.test_normalize = True
        self.beta = 1.0
        self.criterion  = torch.nn.CrossEntropyLoss()

        print('Initialised flatNCE with ESS')

    def forward(self, x, label=None): #x.shape = torch.Size([20, 2, 12])
        assert x.size()[1] >= 2

        out_anchor      = torch.mean(x[:,1:,:],1) # out_anchor = torch.Size([20, 20])  #왜 mean을 쓴거지?
        out_positive    = x[:,0,:] #out_postive = torch.Size([20, 20])
        stepsize        = out_anchor.size()[0] #20

        cos_sim_matrix  = F.cosine_similarity(out_positive.unsqueeze(-1),out_anchor.unsqueeze(-1).transpose(0,2)) 
        
        label = torch.from_numpy(numpy.asarray(range(0,stepsize))).cuda() #label.shape = torch.Size([20])
        nloss = 0
        ess = 0
        for i in range(0, stepsize):
            tempsim = self.beta * (torch.concat([cos_sim_matrix[:,i][:i], cos_sim_matrix[:,i][i+1:]]) - cos_sim_matrix[i,i])
            temploss = torch.logsumexp(tempsim,0)
            ess += 1.0 / (stepsize * torch.sum(torch.square(torch.sigmoid(tempsim))))
            nloss += torch.exp(temploss - temploss.detach())
        nloss = nloss / stepsize
        ess = ess / stepsize
        # if ess > (1.0/stepsize):
        if ess > 0.2:
            self.beta = (1-0.01) * self.beta
        else:
            self.beta = (1+0.01) * self.beta
        # print(" ess: %f, beta: %f" % (ess.item(), self.beta))

        prec1, _    = accuracy(cos_sim_matrix.detach().cpu(), label.detach().cpu(), topk=(1, 2))
        # pdb.set_trace()

        return nloss, prec1