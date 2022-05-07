import argparse
import os
import cv2
import random
import shutil
import time
import warnings

from enum import Enum
from pathlib import Path

import configure
import log_record
import data_argumentation
import ResNet
import concat_backbone
import finetune
from lr_scheduler import VerboseToLogReduceLROnPlateau

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
# from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from tqdm import trange


# print(model)
# create check weight model 用來確認 concat 的 backbone 有被 fix
model1 = ResNet.resnet50(0)
model2 = ResNet.resnet50(0)
# print(ck_model)
checkpoint1 = torch.load('E:/Model/torch/office-31/amazon/none/batch16/resnet50pretrain_fix_backbone/SGD/ReduceLROnPlateau/model_best.pth.tar')
checkpoint2 = torch.load('E:/Model/torch/office-31/amazon/none/batch16/resnet50fix_backbone_tolayer4/SGD/ReduceLROnPlateau/model_best.pth.tar')
model1.load_state_dict(checkpoint1['state_dict'])
model2.load_state_dict(checkpoint2['state_dict'])
ck_model1 = model1
ck_model2 = model2
# ck_model1 = torch.nn.Sequential(*list(model1.children())[:-1])
# ck_model2 = torch.nn.Sequential(*list(model1.children())[:-1])

names1 = []
parameters1 = []
for name, param in ck_model1.layer1.named_parameters():
    names1.append(name)
    parameters1.append(param.cpu().detach().numpy())

names2 = []
parameters2 = []
for name, param in ck_model2.layer1.named_parameters():
    names2.append(name)
    parameters2.append(param.cpu().detach().numpy())

# print(model1)
# get all layer names and weights
if len(names1) != len(names2):
    raise RuntimeError
if len(parameters1) != len(parameters2):
    raise RuntimeError
for i in range(len(names1)):
    if names1[i] != names2[i]:
        raise RuntimeError
for i in range(len(parameters1)):
    compare = parameters1[i] == parameters2[i]
    if not compare.all():
        raise RuntimeError
print('all weights the same')