import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# Read the input image
img = cv2.imread('test7.jpg')
img2 = cv2.imread('test7.jpg')
img3 = mpimg.imread('test7.jpg')

img_h, img_w, img_c = img.shape
print("img size : ", img_w, img_h)

# pixel average
# Red = []
# Green = []
# Blue = []

# for t in img3:
#     for p in t:
#         Red.append(p[0])
#         Green.append(p[1])
#         Blue.append(p[2])

# R_avg = sum(Red) / len(Red)
# G_avg = sum(Green) / len(Green)
# B_avg = sum(Blue) / len(Blue)
  
# Convert into grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

gray = cv2.copyMakeBorder(gray, img_h, img_h, img_w, img_w, cv2.BORDER_CONSTANT, value=[110,110,110])

img = cv2.copyMakeBorder(img, img_h, img_h, img_w, img_w, cv2.BORDER_CONSTANT, value=[110,110,110])

# test_img_h, test_img_w, test_img_c = img.shape
# print("padding_img size : ", test_img_w, test_img_h)

# cv2.imshow("gray", gray)
# cv2.waitKey()
  
# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
  
# Detect faces
# Gray / scale factor / minNeighbours
faces = face_cascade.detectMultiScale(gray, 1.1, 10)
  
# Draw rectangle around the faces and crop the faces
print(len(faces))

for (x, y, w, h) in faces:
    expend_size_w = w//6
    expend_size_h = h//6
    
    if x-expend_size_w < 0 or y-expend_size_h < 0:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 1)
        faces = img[y : y+h, x : x+w]
    else:
        cv2.rectangle(img, (x-expend_size_w, y-expend_size_h), (x+w+expend_size_w, y+h+expend_size_h), (0, 0, 255), 1)
        faces = img[y-expend_size_h : y+h+expend_size_h, x-expend_size_w : x+w+expend_size_w]
    
    # original
    cv2.rectangle(img2, (x, y), (x+w, y+h), (0, 0, 255), 1)
    
    # crop face
    
    
# crop image
# need this img
cv2.imshow("face",faces)
cv2.imwrite('face.jpg', faces)


print("original : ", x,y,w,h)
print("larger box : ", x-expend_size_w, y-expend_size_h, w+expend_size_w*2, h+expend_size_h*2)
      
# Display the output
# box image
cv2.imwrite('detcted.jpg', img)
cv2.imshow('box_lager', img)
cv2.imshow('original', img2)
cv2.waitKey()