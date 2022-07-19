#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import time, pdb, numpy
from utils import accuracy

class LossFunction(nn.Module):
	def __init__(self, fine_tunning, nOut, nClasses, **kwargs):
	    super(LossFunction, self).__init__()

	    self.test_normalize = True
	    self.fine_tunning = fine_tunning
	    
	    self.criterion  = torch.nn.CrossEntropyLoss()
	    if self.fine_tunning == True:
	    	self.fc 		= nn.Linear(nOut,nClasses) # (nmaps, classes)

	    self.n_maps = n_maps
	    self.nClasses = nClasses

	    print('Initialised Softmax Loss')

	def forward(self, x, label=None):
		if self.fine_tunning:
			x 		= self.fc(x)
		nloss   = self.criterion(x, label)
		prec1	= accuracy(x.detach(), label.detach(), topk=(1,))[0]

		return nloss, prec1