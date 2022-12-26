import os, glob
import shutil    
import random
from itertools import combinations

path4 = "vgg-face-2-all/data/test"
directories = glob.glob(path4 + '/*/*')
val = []
cnt = 0
for i in directories:
  cnt = cnt + 1
  if cnt % 50 != 0:
    continue
  print(str(i)[25:])
  val.append(str(i)[25:])

f = open("original_validation_list.csv", 'w')
      
val2 = list(combinations(val, 2))

cnt0 = 0
cnt1 = 0

#print(val2)

for i in val2:
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