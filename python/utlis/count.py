import os, glob
import shutil

folderlen = len(os.listdir("test"))
print(folderlen)


# file_path = "./koreaface_joon/val"
# count = 0
# for f in os.listdir(file_path):
#     if f == ".DS_Store":
#         continue
#     print(f)
#     folderlen = len(os.listdir(file_path+ "/" + f))
#     print(folderlen)
#     count = count + folderlen
# print("count : ", count)
    