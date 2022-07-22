import os
import tarfile

directory_path = 'E:/Dataset/imagenet/ILSVRC2012_img_train/'
walk_generator = os.walk(directory_path)
files = next(walk_generator)[2]
count = 0
for file in files:
    file_name, _ = os.path.splitext(file)
    os.mkdir(directory_path+file_name)
    tar = tarfile.open(directory_path+file, 'r')
    tar.extractall(directory_path+file_name)
    count = count + 1
    print(file_name+' done. '+str(count)+'/1000')
