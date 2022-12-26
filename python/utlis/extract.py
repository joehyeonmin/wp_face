import os
import urllib.request
from urllib import request
import requests
import urllib


file_path = "./vgg_face_dataset/files"

# print(dir_path)

for f in os.listdir(file_path):
    # human_path = open(d, 'r')
    # line = human_path.readline()
    print(f)
    
    
    f = open(file_path + "/" + f, 'r')
    line = None
    cnt = 0
    cnt = cnt + 1
    while line != '':
        line = f.readline()
        
        
        tmp = str(line)
        
        text_split = tmp.split(" ")
        
        print(text_split)
        
        name = text_split[0]
        url = text_split[1]
        
        try:
            img_data = requests.get(url).content
        except:
            continue
        
        folder_name = 'id' + format(cnt, '05')

        if not os.path.isdir(folder_name):
            os.makedirs(folder_name)
            

        jpg_name = name[3:] + '.jpg'
        
        with open(folder_name + "/" + jpg_name, 'wb') as handler:
            handler.write(img_data)
            
        # print(name, url)


    
    
    
