import os
from shutil import copyfile
import cv2


file_path = "vgg-face-2-all/data/train"

for d in os.listdir(file_path):
    #print(d)
    if d == ".DS_Store":
        continue
    
    if not os.path.isdir("vgg2_crop_train/" + d):
        os.makedirs("vgg2_crop_train/" + d) 
    
    for f in os.listdir(file_path + "/" + d):
        #print(file_path + "/" + d + "/" + f)
        img = cv2.imread(file_path + "/" + d + "/" + f)
        img_h, img_w, img_c = img.shape
        
        # Convert into grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.copyMakeBorder(gray, img_h, img_h, img_w, img_w, cv2.BORDER_CONSTANT, value=[110,110,110])
        img = cv2.copyMakeBorder(img, img_h, img_h, img_w, img_w, cv2.BORDER_CONSTANT, value=[110,110,110])
        
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
        # Gray / scale factor / minNeighbours
        faces = face_cascade.detectMultiScale(gray, 1.1, 10)
        
        for (x, y, w, h) in faces:
            expend_size_w = w//2
            expend_size_h = h//2
            
            #print(expend_size_w, expend_size_h)

            # scale
            
            if x-expend_size_w < 0 or y-expend_size_h < 0:
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 1)
                faces = img[y : y+h, x : x+w]
            else:
                cv2.rectangle(img, (x-expend_size_w, y-expend_size_h), (x+w+expend_size_w, y+h+expend_size_h), (0, 0, 255), 1)
                faces = img[y-expend_size_h : y+h+expend_size_h, x-expend_size_w : x+w+expend_size_w]

            print("vgg2_crop_train/" + d + "/" + f)
            if not os.path.isfile("vgg2_crop_train/" + d + "/" + f):
                cv2.imwrite("vgg2_crop_train/" + d + "/" + f, faces)