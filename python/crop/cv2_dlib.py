# mac에서 실행 불가능(home brew 설치해야함)

import dlib
import cv2
  
face_detector = dlib.get_frontal_face_detector()
img = cv2.imread("test.jpeg")
faces = face_detector(img)

for f in faces:
    cv2.rectangle(img, (f.left(), f.top()), (f.right(), f.bottom()), (0,0,255),2)
    
win = dlib.image_window()
win.set_image(img)
win.add_overlay(faces)
cv2.imwrite("output.jpg", img)