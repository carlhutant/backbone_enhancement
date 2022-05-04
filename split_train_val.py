import os
import math
import random
import shutil
import configure
import numpy as np
from pathlib import Path

dataset = 'office-31'
random.seed(486)
source_dir = Path('{}/{}/img/no_split/'.format(configure.dataset_dir, dataset))
target_dir = Path('{}/{}/img/'.format(configure.dataset_dir, dataset))

root_walk_generator = os.walk(source_dir)
_, domain, _ = next(root_walk_generator)
# count = []
for d in domain:
    train_dir = target_dir.joinpath(d).joinpath('train')
    val_dir = target_dir.joinpath(d).joinpath('val')
    adv_dir = source_dir.joinpath(d)
    domain_walk_generator = os.walk(adv_dir)
    _, category, _ = next(domain_walk_generator)
    # cat_count = []
    for c in category:
        os.makedirs(train_dir.joinpath(c))
        os.makedirs(val_dir.joinpath(c))
        cat_dir = adv_dir.joinpath(c)
        category_walk_generator = os.walk(cat_dir)
        _, _, img = next(category_walk_generator)
        img_count = len(img)
        random_list = []
        for i in range(img_count):
            random_list.append(True)
        val_num = math.ceil(img_count * 0.2)
        for i in range(val_num):
            random_list[i] = False
        random.shuffle(random_list)
        for i in range(img_count):
            if random_list[i]:
                shutil.copyfile(cat_dir.joinpath(img[i]), train_dir.joinpath(c).joinpath(img[i]))
            else:
                shutil.copyfile(cat_dir.joinpath(img[i]), val_dir.joinpath(c).joinpath(img[i]))
    #     cat_count.append(len(img))
    # count.append(cat_count)
# for i in range(3):
#     print('{}: {}, max{}, min{}'.format(adv[i], sum(count[i]), max(count[i]), min(count[i])))
# for i in range(31):
#     print('{}: {}, {}, {}'.format(cat[i], count[0][i], count[1][i], count[2][i]))

