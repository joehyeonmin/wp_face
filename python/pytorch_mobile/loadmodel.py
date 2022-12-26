from torchvision.models import resnet18, efficientnet_b0, EfficientNet_B0_Weights, ResNet18_Weights
import torch
import urllib.request
import ssl


ssl._create_default_https_context = ssl._create_unverified_context

m2 = torch.jit.load("mobilenet_V3.pt")
m3 = torch.jit.load("model.pt")
# m4 = torch.load("model.pt")
m2.eval()
m3.eval()

print("make : ", m2 , " possible : ", m3)

print("fail")
print(m2.code)
print("success")
print(m3.code)

print("========================================")

print("fail")
print(torch.load("mobilenet_V3.pt"))
print("success")
print(torch.load("model.pt"))