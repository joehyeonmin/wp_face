import torch.nn as nn
import torch.nn.functional as F
import numpy, math, pdb, sys
import time, importlib
from torch.cuda.amp import autocast, GradScaler
import cv2
import glob
from PIL import Image

sys.path.append('detectors')

from detectors import S3FD

DET = S3FD(device='cpu')

image = cv2.imread("test7.jpg")
image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

bboxes = DET.detect_faces(image_np, conf_th=0.9, scales=[0.5])

bsi = 100
print(bboxes[0][0], bboxes[0][2], bboxes[0][1], bboxes[0][3])

sx = int((bboxes[0][0]+bboxes[0][2])/2) + bsi
sy = int((bboxes[0][1]+bboxes[0][3])/2) + bsi
ss = int(max((bboxes[0][3]-bboxes[0][1]),(bboxes[0][2]-bboxes[0][0]))/1.5)

image = numpy.pad(image,((bsi,bsi),(bsi,bsi),(0,0)), 'constant', constant_values=(110,110))

face = image[int(sy-ss):int(sy+ss),int(sx-ss):int(sx+ss)]
face = cv2.resize(face,(240,240))

cv2.imwrite('s3fd_crop.jpg', face)
