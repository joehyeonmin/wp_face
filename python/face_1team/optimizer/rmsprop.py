#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch

def Optimizer(parameters, lr, weight_decay, **kwargs):

	print('Initialised RMSprop optimizer')

	return torch.optim.RMSprop(parameters, lr = lr, weight_decay = weight_decay);
