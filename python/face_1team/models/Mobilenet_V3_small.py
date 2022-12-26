#! /usr/bin/python
# -*- encoding: utf-8 -*-

# acc@1 (on ImageNet-1K)
# 67.668

# acc@5 (on ImageNet-1K)
# 87.402

# num_params
# 2542856


import torchvision

def MainModel(nOut=256, **kwargs):
    
    return torchvision.models.mobilenet_v3_small(num_classes=nOut)
