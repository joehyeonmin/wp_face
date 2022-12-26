import os

pathfolder = ""
files = os.listdir()

for i in files:
    print(i)
    print(i[-3:])
    if i[-3:] == 'JPG':
        print(i)
        os.rename(i, i[0:-3] + "jpg")