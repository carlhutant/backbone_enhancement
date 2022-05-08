import os

log_dir = 'E:\\Model\\torch\\log.txt'
f = open(log_dir, 'r')
print('train\tval')
train_bool = False
val_bool = False
line = f.readline()
while line != '':
    if 'train' in line:
        if val_bool:
            raise RuntimeError
        train_bool = True
        train = line[-8:-2]
    if 'val' in line:
        if not train_bool:
            raise RuntimeError
        val_bool = True
        val = line[-8:-2]
    if train_bool and val_bool:
        print('{}\t{}'.format(train, val))
        train_bool = False
        val_bool = False
    line = f.readline()
a = 0
