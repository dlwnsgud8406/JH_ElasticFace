import os
import pathlib
import random

root = '/home/user3/ElasticFace/train_copy/'

for path, subdirs, files in os.walk(root):
#     print(path)
    for i, name in enumerate(files):
#         print(name)
        extension = name.split(".")[-1].lower()
        print(extension)
        if extension != "jpg":
            continue
        
#         if i == 0:
#             stri = '00'
#         elif 0 < i < 10:
#             stri = '0' + str(i)
#         else:
#             stri = str(i)
        
#         stri = str(random(1,1000))
#         print(stir)
        new_name = f'{os.path.basename(path)}_000{i + 1}.{extension}'
        print(new_name)
        os.rename(os.path.join(path, name), os.path.join(path, new_name))
        
        print(os.path.basename(path))