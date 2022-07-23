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

    def __init__(self, init_w=10.0, init_b=-5.0, **kwargs):
        super(LossFunction, self).__init__()

        self.test_normalize = True
        self.w = nn.Parameter(torch.tensor(init_w))
        self.b = nn.Parameter(torch.tensor(init_b))
        self.criterion  = torch.nn.CrossEntropyLoss()

        self.use_prototype = kwargs['use_prototype']
        if self.use_prototype:
            self.Q_size = kwargs['Q_size']
            self.dim = kwargs['nOut']
            self.num_classes = kwargs['nOut']+1
            self.register_buffer("queue", torch.randn(self.num_classes, self.Q_size, self.dim))
            self.queue = F.normalize(self.queue, dim=2)
            self.register_buffer("queue_ptr", torch.zeros(self.num_classes, dtype=torch.long))
        self.start_epoch = 1

        print('Initialised AngleProto')

    def forward(self, x, epoch=None, classes=None): #x.shape = torch.Size([20, 2, 12])

        assert x.size()[1] >= 2

        out_anchor = x[:,1,:]
        out_positive = x[:,0,:]
        dict_size = out_anchor.size(0)

        if self.use_prototype:
            classes = classes[0]
            self._dequeue_and_enqueue(keys=out_anchor, classes=classes)
            prototype = self.queue[classes].mean(dim=1)

        if epoch < self.start_epoch or ~self.use_prototype:
            cos_sim_matrix  = F.cosine_similarity(out_positive.unsqueeze(-1),out_anchor.unsqueeze(-1).transpose(0,2))
        else:
            cos_sim_matrix  = F.cosine_similarity(out_positive.unsqueeze(-1),prototype.unsqueeze(-1).transpose(0,2)) 
        torch.clamp(self.w, 1e-6)
        cos_sim_matrix = cos_sim_matrix * self.w + self.b
        
        label       = torch.from_numpy(numpy.asarray(range(0,dict_size))).cuda() #label.shape = torch.Size([20])
        nloss       = self.criterion(cos_sim_matrix, label)
        prec1, _    = accuracy(cos_sim_matrix.detach().cpu(), label.detach().cpu(), topk=(1, 2))

        return nloss, prec1

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, classes):
        ptrs = self.queue_ptr[classes]
        self.queue[classes, ptrs, :] = keys
        ptrs = (ptrs + 1) % self.Q_size
        self.queue_ptr[classes] = ptrs



class LossFunction2(nn.Module):

    def __init__(self, init_w=10.0, init_b=-5.0, **kwargs):
        super(LossFunction2, self).__init__()

        self.test_normalize = True
        
        self.w = nn.Parameter(torch.tensor(init_w))
        self.b = nn.Parameter(torch.tensor(init_b))
        self.criterion  = torch.nn.CrossEntropyLoss()

        self.Q_size = 1
        self.dim = kwargs['nOut']
        self.num_classes = kwargs['nOut']+1

        self.register_buffer("queue", torch.randn(self.num_classes, self.Q_size, self.dim))
        self.queue = F.normalize(self.queue, dim=2)
        self.register_buffer("queue_ptr", torch.zeros(self.num_classes, dtype=torch.long))
        self.start_epoch = 11 

        print('Initialised AngleProto')

    def forward(self, x, epoch, classes=None): #x.shape = torch.Size([20, 2, 12])

        classes = classes[0]
        assert x.size()[1] >= 2

        out_anchor = x[:,1,:]
        out_positive = x[:,0,:]
        dict_size = out_anchor.size(0)
        self._dequeue_and_enqueue(keys=out_anchor, classes=classes)
        cos_sim_matrix  = F.cosine_similarity(out_positive.unsqueeze(-1),self.queue.squeeze(dim=1).unsqueeze(-1).transpose(0,2))
        
        torch.clamp(self.w, 1e-6)
        cos_sim_matrix = cos_sim_matrix * self.w + self.b
        # label       = torch.from_numpy(numpy.asarray(range(0,dict_size))).cuda() #label.shape = torch.Size([20])
        nloss       = self.criterion(cos_sim_matrix, classes)
        prec1, _    = accuracy(cos_sim_matrix.detach().cpu(), classes.detach().cpu(), topk=(1, 2))

        return nloss, prec1


    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, classes):
        ptrs = self.queue_ptr[classes]
        self.queue[classes, ptrs, :] = keys
        ptrs = (ptrs + 1) % self.Q_size
        self.queue_ptr[classes] = ptrs