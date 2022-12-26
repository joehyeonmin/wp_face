#! /usr/bin/python
# -*- encoding: utf-8 -*-

# acc@1 (on ImageNet-1K)
# 84.228

# acc@5 (on ImageNet-1K)
# 96.878

# num_params
# 21458488

import torchvision

def MainModel(nOut=256, **kwargs):
    
    return torchvision.models.efficientnet_v2_s(num_classes=nOut)
