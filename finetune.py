import os
import torch
import torch.nn as nn
import ResNet
import configure

from torch import Tensor


# 現在只實作重新訓練 FC
class FineTuneResNet(nn.Module):
    def __init__(self, model_id: int, pretrained: bool) -> None:
        super().__init__()

        self.pretrained = pretrained
        # 建立 models
        if configure.model[model_id] == 'resnet50':
            self.model = ResNet.resnet50(model_id=model_id, pretrained=pretrained)
        elif configure.model[model_id] == 'resnet101':
            self.model = ResNet.resnet101(model_id=model_id, pretrained=pretrained)
        else:
            raise RuntimeError
        if 'fix_backbone' in configure.model_mode[model_id]:
            self.model.requires_grad_(False)
        if 'train_layer4' in configure.model_mode[model_id]:
            self.layer4_trainable()
        elif 'train_layer3' in configure.model_mode[model_id]:
            self.layer3_trainable()
        elif 'train_layer2' in configure.model_mode[model_id]:
            self.layer2_trainable()
        elif 'train_layer1' in configure.model_mode[model_id]:
            self.layer1_trainable()
        elif 'train_fc' in configure.model_mode[model_id]:
            self.fc_trainable()

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

        # self.remove_fc()

    def forward(self, x: Tensor):
        x = self.model(x)
        # x = self.fc(x)
        return x

    def ck_fc(self):
        print(self)
        names = []
        parameters = []
        for name, param in self.named_parameters():
            names.append(name)
            parameters.append(param)
        stop = 1

    def state_dict(self):
        return self.model.state_dict()

    def layer1_trainable(self):
        self.model.layer1.requires_grad_(True)
        self.layer2_trainable()

    def layer2_trainable(self):
        self.model.layer2.requires_grad_(True)
        self.layer3_trainable()

    def layer3_trainable(self):
        self.model.layer3.requires_grad_(True)
        self.layer4_trainable()

    def layer4_trainable(self):
        self.model.layer4.requires_grad_(True)
        self.fc_trainable()

    def fc_trainable(self):
        self.model.fc.requires_grad_(True)
