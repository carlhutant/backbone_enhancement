import os
import torch
import torch.nn as nn
import ResNet
import configure

from torch import Tensor


class FineTuneResNet(nn.Module):
    def __init__(self, model_id: int, pretrained: bool) -> None:
        super().__init__()
        # 建立 models
        # pretrained 是判斷是否使用 imagenet pretrained weight
        # 因為也可改用 local train 好的 weight 做 finetune
        if configure.model[model_id] == 'resnet50':
            self.model = ResNet.resnet50(model_id=model_id, pretrained=pretrained)
        elif configure.model[model_id] == 'resnet101':
            self.model = ResNet.resnet101(model_id=model_id, pretrained=pretrained)
        else:
            raise RuntimeError

        # layer4 最靠近 FC
        # layerX_trainable 會自動呼叫 layerX 跟 FC 之間的所有 layer_trainable
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

    def load(self, path):
        if os.path.isfile(path):
            print("=> loading checkpoint '{}'".format(path))
            if configure.specified_GPU_ID is None:
                checkpoint = torch.load(path)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(configure.specified_GPU_ID)
                checkpoint = torch.load(path, map_location=loc)
            best_acc1 = checkpoint['best_acc1']
            if configure.specified_GPU_ID is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(configure.specified_GPU_ID)
            self.model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})".format(path, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(path))
            raise RuntimeError

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
