import os
from pyexpat import model
import torch
import torch.nn as nn
import ResNet
import configure

from torch import Tensor


class ConcatResNet50(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc_removed = False

        # 建立 models
        if not configure.multi_model:
            raise RuntimeError
        if configure.model1 == 'resnet50':
            self.model1 = ResNet.resnet50(configure.model_mode1)
        elif configure.model1 == 'resnet101':
            self.model1 = ResNet.resnet101(configure.model_mode1)
        else:
            raise RuntimeError
        if configure.model2 == 'resnet50':
            self.model2 = ResNet.resnet50(configure.model_mode2)
        elif configure.model1 == 'resnet101':
            self.model2 = ResNet.resnet101(configure.model_mode2)
        else:
            raise RuntimeError

        if configure.ina_type is None:
            self.model1.requires_grad_(False)
        self.model2.requires_grad_(False)

        self.dropout = torch.nn.Dropout(p=configure.dropout_rate)

        # 建立剩下的 FC
        if configure.ina_type is None:
            fc_channel = 4096
            if configure.model_mode1 == 'half':
                fc_channel -= 1024
            if configure.model_mode2 == 'half':
                fc_channel -= 1024
        else:
            fc_channel = 2048
        self.fc = torch.nn.Linear(fc_channel, configure.class_num)

    def load(self, path1, path2, args):
        load_model_list = []
        if configure.ina_type is None:
            load_model_list.append((self.model1, path1))
        load_model_list.append((self.model2, path2))

        for model, path in load_model_list:
            if os.path.isfile(path):
                print("=> loading checkpoint '{}'".format(path))
                if args.gpu is None:
                    checkpoint = torch.load(path)
                else:
                    # Map model to be loaded to specified single gpu.
                    loc = 'cuda:{}'.format(args.gpu)
                    checkpoint = torch.load(path, map_location=loc)
                best_acc1 = checkpoint['best_acc1']
                if args.gpu is not None:
                    # best_acc1 may be from a checkpoint from a different GPU
                    best_acc1 = best_acc1.to(args.gpu)
                model.load_state_dict(checkpoint['state_dict'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(path, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(path))
                raise RuntimeError

        self.remove_fc()
        # get all layer names and weights
        # names1 = []
        # parameters1 = []
        # for name, param in self.model1.named_parameters():
        #     names1.append(name)
        #     parameters1.append(param.cpu().detach().numpy())
        #
        # names2 = []
        # parameters2 = []
        # for name, param in self.model2.named_parameters():
        #     names2.append(name)
        #     parameters2.append(param.cpu().detach().numpy())
        # stop = 1

    def remove_fc(self):
        if not self.fc_removed:
            self.model1 = torch.nn.Sequential(*list(self.model1.children())[:-1])
            self.model2 = torch.nn.Sequential(*list(self.model2.children())[:-1])
            self.fc_removed = True

    def forward(self, x: Tensor):
        data1, data2 = torch.split(x, [3, 3], dim=1)
        x1 = self.model1(data1)
        x2 = self.model2(data2)
        if configure.ina_type is None:
            x = torch.concat([x1, x2], dim=1)
            x = torch.flatten(x, 1)
            x = self.dropout(x)
            x = self.fc(x)
        else:
            x = torch.flatten(x1, 1)
            x = self.fc(x)

        if configure.ina_type is None:
            return x
        else:
            return x, x1, x2

    def ck_fc(self):
        print(self)
        names = []
        parameters = []
        for name, param in self.named_parameters():
            names.append(name)
            parameters.append(param)
        stop = 1
