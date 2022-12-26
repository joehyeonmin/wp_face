import cv2
from PIL import Image
import torchvision.transforms as transforms
import numpy as np


test_transform = transforms.Compose(
        [ transforms.ToTensor(),
        #  transforms.Resize(256),
        #  transforms.CenterCrop([224,224]),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

print("face")
# im1 = Image.open('face-1.jpg'.format("face")) 
im1 = Image.open('face-1.jpg') 
im2 = Image.open('face.jpg') 
# print(np.array(im1))

img = im1
img = img.resize((256,256),Image.Resampling.NEAREST)
width, height = img.size   # Get dimensions
left = (width - 224)/2
top = (height - 224)/2
right = (width + 224)/2
bottom = (height + 224)/2
# Crop the center of the image
im1 = img.crop((left, top, right, bottom))

inp1 = test_transform(im1)
test = test_transform(im1)
print("tensor1 : ", (inp1))
print("tensor1 shape : ", (inp1).shape)
data1    = inp1.reshape(-1,inp1.size()[-3],inp1.size()[-2],inp1.size()[-1])
print("reshape tensor1 : ", (data1))
# print("comparision : ", torch.eq(data1,test))
print("shape1 : ", data1.shape)
print()

img = im2
img = img.resize((256,256),Image.Resampling.NEAREST)
width, height = img.size   # Get dimensions
left = (width - 224)/2
top = (height - 224)/2
right = (width + 224)/2
bottom = (height + 224)/2
# Crop the center of the image
im2 = img.crop((left, top, right, bottom))


inp2 = test_transform(im2)
print("tensor2 : ", inp2)
print("tensor2 shape : ", (inp2).shape)
data2    = inp2.reshape(-1,inp2.size()[-3],inp2.size()[-2],inp2.size()[-1])
# print("comparision : ", torch.eq(inp2, data2))
print("reshape tensor2 : ", (data2))
print("shape2 : ", data2.shape)


import torch.nn.functional as F

import torch
import torchvision

# model = torchvision.models.mobilenet_v3_small(num_classes=512)
model = torch.jit.load("jit_model.pt")
print(model)

print("reshape data shape : ", data2.shape)
com_feat1 = model(data1)
com_feat2 = model(data2)
print("reshape result1 shape : ", com_feat1.shape)
print("reshape result1 : ", com_feat1)
print()
print("reshape result2 shape : ", com_feat2.shape)
print("reshape result2 : ", com_feat2)

score = F.cosine_similarity(com_feat1, com_feat2)
print("score")
print(score.item())
