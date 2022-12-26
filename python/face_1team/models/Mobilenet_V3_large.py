#! /usr/bin/python
# -*- encoding: utf-8 -*-

# acc@1 (on ImageNet-1K)
# 74.042

# acc@5 (on ImageNet-1K)
# 91.34

# num_params
# 5483032


import torchvision

def MainModel(nOut=256, **kwargs):
    
    return torchvision.models.mobilenet_v3_large(num_classes=nOut)
