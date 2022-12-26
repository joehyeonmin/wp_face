#! /usr/bin/python
# -*- encoding: utf-8 -*-

# acc@1 (on ImageNet-1K)
# 77.618

# acc@5 (on ImageNet-1K)
# 93.698

# num_params
# 2502 8904


import torchvision

def MainModel(nOut=256, **kwargs):
    
    return torchvision.models.resnext50_32x4d(num_classes=nOut)
