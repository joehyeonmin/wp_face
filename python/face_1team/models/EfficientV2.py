#! /usr/bin/python
# -*- encoding: utf-8 -*-

# acc@1 (on ImageNet-1K)
# 85.112

# acc@5 (on ImageNet-1K)
# 97.156

# num_params
# 54139356


import torchvision

def MainModel(nOut=256, **kwargs):
    
    return torchvision.models.efficientnet_v2_m(num_classes=nOut)
