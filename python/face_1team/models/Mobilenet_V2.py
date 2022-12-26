#! /usr/bin/python
# -*- encoding: utf-8 -*-

# acc@1 (on ImageNet-1K)
# 71.878

# acc@5 (on ImageNet-1K)
# 90.286

# num_params
# 3504872


import torchvision

def MainModel(nOut=256, **kwargs):
    
    return torchvision.models.mobilenet_v2(num_classes=nOut)
