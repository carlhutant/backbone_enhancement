import os

log_dir = 'E:\\Model\\torch\\log.txt'
f = open(log_dir, 'r')
print('train-val')
train_bool = False
val_bool = False
line = f.readline()
while line != '':
    if 'train' in line:
        if val_bool:
            raise RuntimeError
        train_bool = True
        train = line[-8:-2]
        if train[0] == ':':
            train = train[1:]
    if 'val' in line:
        if not train_bool:
            raise RuntimeError
        val_bool = True
        val = line[-8:-2]
        if val[0] == ':':
            val = val[1:]
    if train_bool and val_bool:
        print('{}-{}'.format(float(train), float(val)))
        train_bool = False
        val_bool = False
    line = f.readline()
a = 0
