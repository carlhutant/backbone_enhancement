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
            # ConcatResNet 會使用自己的 FC，所有建立的 backbone 在 load 完 weight 後會關閉其 FC
            configure.model_mode[i] += '_removeFC'
            if configure.model[i] == 'resnet50':
                self.model_list.append(ResNet.resnet50(i))
            elif configure.model[i] == 'resnet101':
                self.model_list.append(ResNet.resnet101(i))
            else:
                raise RuntimeError
            # 像 INA 在訓練 target backbone 時會 fix
            if 'fix_backbone' in configure.model_mode[i]:
                self.model_list[i].requires_grad_(False)

        if configure.dropout_rate is not None:
            self.dropout = torch.nn.Dropout(p=configure.dropout_rate)

        if configure.ina_type is None:
            # 不使用 INA 時所有 backbone 的 feature 都會傳給 FC
            fc_channel = configure.backbone_out_channel_sum
        else:
            # 使用 INA 時只有第一個 backbone 的 feature 都會傳給 FC
            # 因此注意 target backbone 要放最前面
            fc_channel = configure.backbone_out_channel_list[0]
        self.fc = torch.nn.Linear(fc_channel, configure.class_num)

    def load(self, optimizer, scheduler):
        load_model_list = []
        if len(configure.resume_ckpt_path) == 1:
            # 一般用在 concat backbone 所存的 model
            print('loading entire model...')
            load_model_list.append((self, configure.resume_ckpt_path[0]))
        elif len(configure.resume_ckpt_path) == configure.model_num:
            # 一般用在組合多個單獨訓練或沒訓練的 backbone
            print('loading multiple backbone...')
            for i in range(configure.model_num):
                load_model_list.append((self.model_list[i], configure.resume_ckpt_path[i]))
        else:
            raise RuntimeError
        best_acc1 = 0
        for i in range(len(load_model_list)):
            model, path = load_model_list[i]
            if path is None:
                print("=> model {} train from scratch".format(i))
            elif os.path.isfile(path):
                print("=> loading checkpoint '{}'".format(path))
                if configure.specified_GPU_ID is None:
                    checkpoint = torch.load(path)
                else:
                    # Map model to be loaded to specified single gpu.
                    loc = 'cuda:{}'.format(configure.specified_GPU_ID)
                    checkpoint = torch.load(path, map_location=loc)

                # 讀取模型 weight 資訊
                model.load_state_dict(checkpoint['state_dict'])
                if len(configure.resume_ckpt_path) == 1 and not configure.reset_training_history:
                    # 讀取 concat backbone 所存的 model 並且沒有重設 training_history
                    # 可能是 target backbone 訓練一半中斷 要接續訓練
                    # 那就需要再讀取 optimizer, scheduler, epoch, best_acc1 等等
                    optimizer.load_state_dict(checkpoint['optimizer'])
                    scheduler.load_state_dict(checkpoint['scheduler'])
                    configure.start_epoch = checkpoint['epoch']
                    best_acc1 = checkpoint['best_acc1']
                    if configure.specified_GPU_ID is not None:
                        # best_acc1 may be from a checkpoint from a different GPU
                        best_acc1 = best_acc1.to(configure.specified_GPU_ID)
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(path, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(path))
                raise RuntimeError
        return best_acc1

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
        # 將擴增過 channel 的 input data 分割給各 backbone
        data = torch.split(x, configure.input_channel_list, dim=1)

        # 接收各 backbone 的輸出
        feature = []
        for i in range(configure.model_num):
            feature.append(self.model_list[i](data[i]))

        # 不使用 INA 時所有 backbone 的 feature 都會傳給 FC
        if configure.ina_type is None:
            x = torch.concat(feature, dim=1)
            x = torch.flatten(x, 1)
            if configure.dropout_rate is not None:
                x = self.dropout(x)
            x = self.fc(x)
            return x

        if configure.dropout_rate is not None:
            feature[0] = self.dropout(feature[0])
        # 使用 INA 時只有第一個 backbone 的 feature 都會傳給 FC
        # 因此注意 target backbone 要放最前面
        predict = self.fc(feature[0])

        # 準備 cycle consistency 所需的 backbone feature
        # 根據 configure.loss_pair_list 直接排好一個一個 pair
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

    # 原先用來在 debug 模式下確認 backbone 有無 fix, channel 是否符合預期等
    def ck_fc(self):
        print(self)
        names = []
        parameters = []
        for name, param in self.named_parameters():
            names.append(name)
            parameters.append(param)
        stop = 1
