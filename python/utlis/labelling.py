import os, glob
from shutil import copyfile


orig_path = "./test"
save_path = "./koreaface"

# print(dir_path)
dcnt = 0
for d in os.listdir(orig_path):
    dcnt = dcnt+1
    print(d)
    
    folder_name = 'id' + format(dcnt, '05')
    
    if not os.path.exists(orig_path+"/"+folder_name):
        os.makedirs(orig_path+"/"+folder_name)

    
    files = glob.glob(orig_path+"/"+d+'/*.jpg') + glob.glob(orig_path+"/"+d+'/*.png') + glob.glob(orig_path+"/"+d+'/*.jpeg')
    #print(files)
    fcnt = 0
    for f in files:
        fcnt = fcnt+1
        ###########
        # crop
        ###########
        
        print(f)
        
        copyfile(f,orig_path+"/"+folder_name+"/"+format(fcnt, '05')+".jpg")
       
    
    
    
    