#! /usr/bin/python
# -*- encoding: utf-8 -*-

# acc@1 (on ImageNet-1K)
# 69.758

# acc@5 (on ImageNet-1K)
# 89.078

# num_params
# 1168 9512


import torchvision

def MainModel(nOut=256, **kwargs):
    
    return torchvision.models.resnet18(num_classes=nOut)
