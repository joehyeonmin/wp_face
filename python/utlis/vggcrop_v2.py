import os
from shutil import copyfile
import cv2
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

DET = S3FD(device='cuda')

file_path = "vggface2/train"

for d in os.listdir(file_path):
    #print(d)
    if d == ".DS_Store":
        continue
    
    if not os.path.isdir("vggface2/" + d):
        os.makedirs("vggface2_crop_s3fd/train/" + d) 
    
    for f in os.listdir(file_path + "/" + d):
        print(file_path + "/" + d + "/" + f)
        image = cv2.imread(file_path + "/" + d + "/" + f)
        image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        try:
            bboxes = DET.detect_faces(image_np, conf_th=0.9, scales=[0.5])
        except:
            continue
        
        if len(bboxes) == 0:
            continue

        bsi = 100
        # print(bboxes[0][0], bboxes[0][2], bboxes[0][1], bboxes[0][3])

        sx = int((bboxes[0][0]+bboxes[0][2])/2) + bsi
        sy = int((bboxes[0][1]+bboxes[0][3])/2) + bsi
        ss = int(max((bboxes[0][3]-bboxes[0][1]),(bboxes[0][2]-bboxes[0][0]))/1.5)

        image = numpy.pad(image,((bsi,bsi),(bsi,bsi),(0,0)), 'constant', constant_values=(110,110))

        face = image[int(sy-ss):int(sy+ss),int(sx-ss):int(sx+ss)]
        
        try:
            face = cv2.resize(face,(240,240))
        except:
            continue

        print("vggface2/" + d + "/" + f)
        if not os.path.isfile("vggface2_crop_s3fd/train/" + d + "/" + f):
            cv2.imwrite("vggface2_crop_s3fd/train/" + d + "/" + f, face)