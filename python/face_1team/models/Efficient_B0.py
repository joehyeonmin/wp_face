#! /usr/bin/python
# -*- encoding: utf-8 -*-

# acc@1 (on ImageNet-1K)
# 77.692

# acc@5 (on ImageNet-1K)
# 93.532

# num_params
# 528 8548


import torchvision

def MainModel(nOut=256, **kwargs):
    
    return torchvision.models.efficientnet_b0(num_classes=nOut)
