import os
import torch
import torch.nn as nn
import ResNet
import configure

from torch import Tensor


class FineTuneResNet(nn.Module):
    def __init__(self, model_mode: str) -> None:
        super().__init__()
        self.fc_removed = False
        self.model_mode = model_mode

        # 建立 models
        if configure.multi_model:
            raise RuntimeError
        if 'resnet50' in configure.model1:
            self.model = ResNet.resnet50(model_mode)
        elif 'resnet101' in configure.model1:
            self.model = ResNet.resnet101(model_mode)
        else:
            raise RuntimeError

    def load(self, path, args):
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
            self.model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})".format(path, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(path))
            raise RuntimeError
        self.model.requires_grad_(False)
        self.remove_fc()

    def remove_fc(self):
        # 建立剩下的 FC
        if not self.fc_removed:
            if self.model_mode == 'half':
                fc_channel = 1024
            else:
                fc_channel = 2048
            self.model.fc = torch.nn.Linear(fc_channel, configure.class_num)
            self.fc_removed = True

    def forward(self, x: Tensor):
        x = self.model(x)
        return x
