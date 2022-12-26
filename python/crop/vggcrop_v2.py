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

DET = S3FD(device='cpu')

file_path = "koreaface3"
count = 1

for d in os.listdir(file_path):
    print(d)
    if d == ".DS_Store":
        continue
    
    if not os.path.isdir("koreaface_1team/" + d):
        os.makedirs("koreaface_1team/id" + '{0:05}'.format(count))
 

    
    count_file = 1
    for f in os.listdir(file_path + "/" + d):
        print("start : ", f)
        
        if f[-3:] != "jpg" or f == ".jpg":
            continue
        
        try:
            image = cv2.imread(file_path + "/" + d + "/" + f)
            image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except:
            continue

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

        # print("end : ", "vggface2/" + d + "/" + f)
        # print("end2 : ", "koreaface_1team/id" + '{0:05}'.format(count) + "/" + '{0:03}'.format(count_file) + ".jpg")
        if not os.path.isfile("koreaface_1team/id" + '{0:05}'.format(count) + "/" + '{0:03}'.format(count_file) + ".jpg"):
            cv2.imwrite("koreaface_1team/id" + '{0:05}'.format(count) + "/" + '{0:03}'.format(count_file) + ".jpg", face)
            
        count_file = count_file + 1
            
    count = count + 1