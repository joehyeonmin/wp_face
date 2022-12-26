import cv2
from time import sleep
import requests
from PIL import Image, ImageDraw
import pickle

cam_port = 0

command = input("Input enroll or query : ")
url = 'http://192.168.100.92:10015/' + command

cam = cv2.VideoCapture(cam_port)
if (cam.isOpened() == False):
    print("Unable to read camera feed")

imgid = 0
while (True):
    print("user capture start")
    result, image = cam.read()
    print("user capture end")

    if result == True:
        #cv2.imshow("GFG", image)
        pass
    else:
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    fn = f'img/image_{imgid}.jpg'
    
    cv2.imwrite(fn, image, [cv2.IMWRITE_PNG_COMPRESSION, 7])  # 0 ~9, 압축율
    # files = {'media': open(fn, 'rb')}

    fnpkl = f'img/image_{imgid}.pkl'
    img = Image.open(fn)

    if command == "enroll":
        cv2.imshow('GFG', image)
        iname = input("input name(change query with m) : ")
        if iname == 'm':
            url = 'http://192.168.100.92:10015/query'
            command = "query"
            continue
        
        tmp = requests.post(url, pickle.dumps({'img':img, 'name':iname}))
        print(tmp.text)
        continue
    else:
        iname = 1111
        # post
        tmp = requests.post(url, pickle.dumps({'img':img, 'name':iname}))
        # print(tmp.text)
        print("post success")
        if tmp.text == "fail":
            print("face detect fail")
            cv2.imshow('GFG', image)
            continue

        s = tmp.text
        s = s.split()
        # for i in s:
        #     print(i)
        #     print(type(i))
        
        print("pred : " + s[4][12:-3] + " per : " + s[5])
        
        print("rectangle test : ", (int(float(s[0])), int(float(s[1])), int(float(s[2])), int(float(s[3]))))
        
        #cv2.rectangle(image, (730, 550), (1169, 995), (255,0,0), 3)
        cv2.rectangle(image, ((int(float(s[0]))), int(float(s[1]))), ((int(float(s[2]))), int(float(s[3]))), (255,0,0), 3)
        
        text_pos = (int(float(s[0])), int(float(s[1])))
        font=cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, s[4][12:-2] +  " per : " + s[5], text_pos, font, 1,(255,0,0), 2)
        cv2.namedWindow("GFG",0);
        cv2.resizeWindow("GFG", 600, 400);
        cv2.imshow('GFG', image)
        
        imgid += 1
        sleep(0.001)
    
cam.release()
cv2.destroyWindow("GFG")