import os, glob
import shutil


folderlen = len(os.listdir("test2"))
print(folderlen)


# shutil.rmtree('./train/')
# shutil.rmtree('./validation/')

if not os.path.isdir('train'):
  os.makedirs("train")

if not os.path.isdir('validation'):
  os.makedirs("validation")

flen = 0

pathfolder = "test2"
directories = glob.glob(pathfolder + '/*')

shutil.rmtree('./train/')
shutil.rmtree('./validation/')

for i in directories:
  print(i)
  if flen < int(folderlen*0.5):
    shutil.move(i,"train")
    flen += 1
  else:
    shutil.move(i,"validation")
    
    
import random

path4 = "validation/"
directories = glob.glob(path4 + '/*/*')
val = []

for i in directories:
  print(str(i))
  val.append(str(i)[11:])


f = open("Validation_list.txt", 'w')

from itertools import combinations       
val = list(combinations(val, 2))

cnt0 = 0
cnt1 = 0

for i in val:
  idx0 = i[0].find("/")
  #print(i[0][0:idx0])

  idx1 = i[1].find("/")
  #print(i[1][0:idx1])

  rand = random.randrange(1,30)
  

  if i[0][0:idx0] == i[1][0:idx1]:
    if cnt0 >= cnt1:
      f.write("1" + "," +  str(i[0]) + "," +  str(i[1]) + "\n")
      cnt1 += 1
  else:
    if cnt0 <= cnt1 and rand == 10:
      f.write("0" + "," +  str(i[0]) + "," +  str(i[1]) + "\n")
      cnt0 += 1