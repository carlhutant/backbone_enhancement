import os
from pyexpat import model
import torch
import torch.nn as nn
import ResNet
import configure

from torch import Tensor


class ConcatResNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # 建立 models
        if not configure.model_num > 1:
            raise RuntimeError
        self.model_list = torch.nn.ModuleList()
        for i in range(configure.model_num):
            configure.model_mode[i] += '_removeFC'
            if configure.model[i] == 'resnet50':
                self.model_list.append(ResNet.resnet50(i))
            elif configure.model[i] == 'resnet101':
                self.model_list.append(ResNet.resnet101(i))
            else:
                raise RuntimeError
            if 'fix_backbone' in configure.model_mode[i]:
                self.model_list[i].requires_grad_(False)

        if configure.ina_type is None:
            self.dropout = torch.nn.Dropout(p=configure.dropout_rate)

        # 建立剩下的 FC
        if configure.ina_type is None:
            fc_channel = configure.output_channel
        else:
            fc_channel = configure.output_channel_list[0]
        self.fc = torch.nn.Linear(fc_channel, configure.class_num)

    def load(self, args):
        load_model_list = []
        if len(configure.resume_ckpt_path) == 1:
            print('loading entire model...')
            load_model_list.append((self, configure.resume_ckpt_path[0]))
        elif len(configure.resume_ckpt_path) == configure.model_num:
            print('loading multiple backbone...')
            for i in range(configure.model_num):
                load_model_list.append((self.model_list[i], configure.resume_ckpt_path[i]))
        else:
            raise RuntimeError

        for i in range(len(load_model_list)):
            model, path = load_model_list[i]
            if path is None:
                print("model {} training on random parameter".format(i))
            elif os.path.isfile(path):
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

    def forward(self, x: Tensor):
        data = torch.split(x, configure.input_channel_list, dim=1)
        feature = []
        for i in range(configure.model_num):
            feature.append(self.model_list[i](data[i]))
        if configure.ina_type is None:
            x = torch.concat(feature, dim=1)
            x = torch.flatten(x, 1)
            x = self.dropout(x)
            x = self.fc(x)
            return x

        predict = self.fc(feature[0])

        feature_pair_list = []
        for pair in configure.loss_pair_list:
            feature_pair_list.append((feature[pair['model'][0]][..., pair['start'][0]:pair['end'][0]+1],
                                      feature[pair['model'][1]][..., pair['start'][1]:pair['end'][1]+1]))
        return predict, feature_pair_list
        # if configure.ina_type == 'full':
        #     return x, x1, x2
        # elif configure.ina_type == 'half':
        #     if x1.size(1) != x2.size(1) * 2:
        #         raise RuntimeError
        #     split_x1 = torch.split(x1, [x2.size(1), x2.size(1)], dim=1)[0]
        #     return x, split_x1, x2
        # elif configure.ina_type == 'fc_only':
        #     return x, x1, x2
        # else:
        #     raise RuntimeError

    def ck_fc(self):
        print(self)
        names = []
        parameters = []
        for name, param in self.named_parameters():
            names.append(name)
            parameters.append(param)
        stop = 1
