#! /usr/bin/python
# -*- encoding: utf-8 -*-

# acc@1 (on ImageNet-1K)
# 80.032

# acc@5 (on ImageNet-1K)
# 95.048

# num_params
# 3938 1472


import torchvision

def MainModel(nOut=256, **kwargs):
    
    return torchvision.models.regnet_y_8gf(num_classes=nOut)
