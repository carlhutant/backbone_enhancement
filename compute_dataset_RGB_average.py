import configure
import os
import cv2
import numpy as np
# import matplotlib.pyplot as plt
from pathlib import Path


# Dataset config #
dataset = 'AWA2'  # AWA2, imagenet
datatype = 'img'  # img, tfrecord, npy
data_advance = 'none'   # color_diff_121, none, color_diff_121_abs
data_usage = 'train'

# Directory set
dataset_dir = configure.dataset_dir
target_directory = '{}/{}/{}/{}/{}/'.format(dataset_dir, dataset, datatype, data_advance, data_usage)

walk_generator = os.walk(target_directory)
root, directory, _ = next(walk_generator)
total_mean_sum = np.zeros(configure.input_channel_list[0], dtype=float)
total_std_sum = np.zeros(configure.input_channel_list[0], dtype=float)
# rgb_count = np.zeros((configure.channel, 256), dtype=np.int64)
image_count = 0
# pixel_count = 0
for d in directory:
    # print(d)
    walk_generator2 = os.walk(root + d)
    flies_root, _, files = next(walk_generator2)
    for file in files:
        image = cv2.imread(str(Path(flies_root).joinpath(file)))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.array(image, dtype=float)
        # height, width, _ = array.shape

        # for x in range(width):
        #     for y in range(height):
        #         for color in range(configure.channel):
        #             rgb_count[color, array[y, x, color]] += 1
        # image_count += 1
        # print(image_count)
        # if image_count == 70:
        #     x = np.arange(256)
        #     plt.bar(x, rgb_count[0])
        #     plt.show()
        #     pause = 1
        # pixel_count = pixel_count + height * width
        total_mean_sum += image.mean((0, 1), dtype=float)
        total_std_sum += image.std(axis=(0, 1), dtype=float)
        image_count += 1
        print(image_count)
total_mean = total_mean_sum/image_count
total_std = total_std_sum/image_count
print(total_mean)
print(total_std)
a = 0
