# 原先範例使用的
import argparse
import os
import random
import shutil
import time
import warnings
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
from torch.optim.lr_scheduler import StepLR
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

# 額外新增
import cv2
import numpy as np
from pathlib import Path
from tqdm import trange

# 自行定義的函式
import configure
import log_record
import data_argumentation
import ResNet
import concat_backbone
import finetune
from lr_scheduler import VerboseToLogReduceLROnPlateau

# 68 種原先可使用的 backbone names
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

best_acc1 = 0
best_acc1_early_stop = 0
count_early_stop = 0


def main():
    # 保存 configure.py  紀錄訓練時的參數設定，同時避免訓練好的模型在搬移前被新的訓練覆蓋
    if not configure.evaluate_only:
        log_rec = log_record.LogRecoder(configure.resume)
        if os.path.exists(Path(configure.ckpt_dir).joinpath('configure.py')):
            raise RuntimeError
        else:
            shutil.copyfile(Path(configure.code_dir).joinpath('configure.py'),
                            Path(configure.ckpt_dir).joinpath('configure.py'))
    else:
        log_rec = None
    # args = parser.parse_args()

    # 初始化模型參數用的隨機碼，不確定餵資料時是否有用到，其實也需要測試是否 seed 固定後模型的初始猜書都是固定的
    if configure.random_seed is not None:
        random.seed(configure.random_seed)
        torch.manual_seed(configure.random_seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if configure.specified_GPU_ID is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    ngpus_per_node = torch.cuda.device_count()

    # 測試模型在各 domain 的效果
    if configure.evaluate_all_domain:
        if configure.dataset == ['AWA2', 'imagenet', 'inat2021']:
            order_list = [[0, 1, 2], [0, 2, 1], [1, 0, 2],
                          [1, 2, 0], [2, 0, 1], [2, 1, 0]]
            for order in order_list:
                configure.rgb_swap_order = order
                print('order:{}'.format(configure.rgb_swap_order))
                main_worker(ngpus_per_node, log_rec)
        elif configure.dataset == 'office-31':
            domain_list = ['amazon', 'dslr', 'webcam']
            for domain in domain_list:
                configure.domain = domain
                print('domain:{}'.format(configure.domain))
                main_worker(ngpus_per_node, log_rec)
        elif configure.dataset == 'OfficeHome':
            domain_list = ['Art', 'Clipart', 'Product', 'Real World']
            for domain in domain_list:
                configure.domain = domain
                print('domain:{}'.format(configure.domain))
                main_worker(ngpus_per_node, log_rec)
        else:
            raise RuntimeError
    else:
        # Simply call main_worker function
        main_worker(ngpus_per_node, log_rec)


def main_worker(ngpus_per_node, log_rec):
    global best_acc1
    global best_acc1_early_stop
    global count_early_stop

    if configure.specified_GPU_ID is not None:
        print("Use GPU: {} for training".format(configure.specified_GPU_ID))

    # create model
    # 總共幾個 backbone 接在一起，兩個以上都由 concat_backbone 的函式處理
    if configure.model_num < 1:
        raise RuntimeError
    elif configure.model_num > 1:
        model = concat_backbone.ConcatResNet()
    else:
        if 'resnet' in configure.model[0]:
            if 'fix_backbone' in configure.model_mode[0]:
                if 'pretrain' in configure.model_mode[0]:
                    pretrained = True
                else:
                    pretrained = False
                model = finetune.FineTuneResNet(model_id=0, pretrained=pretrained)
            elif configure.model[0] == 'resnet50':
                model = ResNet.resnet50(model_id=0)
            elif configure.model[0] == 'resnet101':
                model = ResNet.resnet101(model_id=0)
            else:
                raise RuntimeError
        else:
            # 目前只支援 resnet50, 101
            raise RuntimeError
            # if args.pretrained:
            #     print("=> using pre-trained model '{}'".format(args.arch))
            #     model = models.__dict__[args.arch](pretrained=True)
            # else:
            #     print("=> creating model '{}'".format(args.arch))
            #     model = models.__dict__[args.arch]()

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif configure.specified_GPU_ID is not None:
        torch.cuda.set_device(configure.specified_GPU_ID)
        model = model.cuda(configure.specified_GPU_ID)
    # 目前設計成需指定單一 GPU
    else:
        raise RuntimeError

    # print(model)
    # for name, param in model.named_parameters():
    #     a = 0

    # define loss function (criterion), optimizer, and learning rate scheduler
    criterion = {'CrossEntropy': nn.CrossEntropyLoss().cuda(configure.specified_GPU_ID),
                 'MSE': nn.MSELoss().cuda(configure.specified_GPU_ID)}

    optimizer = torch.optim.SGD(model.parameters(), configure.lr,
                                momentum=configure.momentum,
                                weight_decay=configure.weight_decay)

    # LR scheduler
    # ReduceLROnPlateau
    scheduler = VerboseToLogReduceLROnPlateau(optimizer,
                                              mode='max',
                                              factor=configure.factor,
                                              patience=configure.patience,
                                              threshold=configure.threshold,
                                              verbose=True,
                                              log_rec=log_rec)
    # # Sets the learning rate to the initial LR decayed by 10 every 30 epochs
    # scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    # resume from a checkpoint
    if configure.resume:
        if configure.model_num > 1:  # 使用 concat backbone
            model.load(optimizer, scheduler)
        elif configure.model_num == 1 and 'fix' in configure.model_mode[0]:  # 使用 finetune
            model.load(configure.resume_ckpt_path[0])
        else:  # 使用 ResNet
            if os.path.isfile(configure.resume_ckpt_path[0]):
                print("=> loading checkpoint '{}'".format(configure.resume_ckpt_path[0]))
                if configure.specified_GPU_ID is None:
                    checkpoint = torch.load(configure.resume_ckpt_path[0])
                else:
                    # Map model to be loaded to specified single gpu.
                    loc = 'cuda:{}'.format(configure.specified_GPU_ID)
                    checkpoint = torch.load(configure.resume_ckpt_path[0], map_location=loc)
                configure.start_epoch = checkpoint['epoch']+1
                best_acc1 = checkpoint['best_acc1']
                if configure.specified_GPU_ID is not None:
                    # best_acc1 may be from a checkpoint from a different GPU
                    best_acc1 = best_acc1.to(configure.specified_GPU_ID)
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                scheduler.load_state_dict(checkpoint['scheduler'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(configure.resume_ckpt_path[0], checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(configure.resume_ckpt_path[0]))

    # # print(model)
    # # create check weight model 用來確認 concat 的 backbone 有被 fix
    # ck_model = ResNet.resnet50('none')
    # # print(ck_model)
    # checkpoint = torch.load('E:/Download/old_model_best.pth.tar')
    # ck_model.load_state_dict(checkpoint['state_dict'])
    # ck_model = torch.nn.Sequential(*list(ck_model.children())[:-1])
    #
    # names2 = []
    # parameters2 = []
    # for name, param in ck_model.named_parameters():
    #     names2.append(name)
    #     parameters2.append(param.cpu().detach().numpy())
    #
    # print(model)
    # # get all layer names and weights
    # names = []
    # parameters = []
    # for name, param in model.backbone.named_parameters():
    #     names.append(name)
    #     parameters.append(param.cpu().detach().numpy())
    #
    # if len(names) != len(names2):
    #     raise RuntimeError
    # if len(parameters) != len(parameters2):
    #     raise RuntimeError
    # for i in range(len(names)):
    #     if names[i] != names2[i]:
    #         raise RuntimeError
    # for i in range(len(parameters)):
    #     compare = parameters[i] == parameters2[i]
    #     if not compare.all():
    #         raise RuntimeError
    # print('all weights the same')

    cudnn.benchmark = True

    # Data loading code
    train_dir = os.path.join(configure.data_dir, 'train')
    val_dir = os.path.join(configure.data_dir, 'val')
    # imagenet rgb mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    mean = []
    std = []
    for data_advance in configure.data_advance:
        if data_advance == 'none':
            mean += [0.485, 0.456, 0.406]
            std += [0.229, 0.224, 0.225]
        elif data_advance == 'color_diff_121_abs_3ch':
            mean += [0.043, 0.043, 0.043]
            std += [0.047, 0.047, 0.047]
        elif data_advance == 'color_diff_121_abs_1ch':
            mean += [0.043]
            std += [0.047]
        else:
            raise RuntimeError

    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform_list = [transforms.ToTensor()]
    val_transform_list = [transforms.ToTensor()]

    # 設定 train_transform_list
    if configure.rgb_swap_order is not None:
        train_transform_list.append(data_argumentation.ChannelSwap(configure.rgb_swap_order))
    if configure.data_advance_num > 1:
        if 'color_diff_121_abs_3ch' in configure.data_advance or 'color_diff_121_abs_1ch' in configure.data_advance:
            train_transform_list.append(transforms.RandomResizedCrop(size=(configure.train_crop_h + 2,
                                                                           configure.train_crop_w + 2),
                                                                     scale=(configure.train_resize_area_ratio_min,
                                                                            configure.train_resize_area_ratio_max),
                                                                     ratio=(configure.train_crop_ratio,
                                                                            configure.train_crop_ratio)))
            train_transform_list.append(data_argumentation.AppendDataAdvance())
        else:
            raise RuntimeError
    elif configure.data_advance_num == 1:
        # 若有使用 conv2D 則 crop 時要留 padding 的空間
        if 'color_diff' in configure.data_advance[0]:
            configure.train_crop_h += 2
            configure.train_crop_w += 2

        if configure.train_crop_type == 'random_crop':
            train_transform_list.append(transforms.RandomResizedCrop(size=(configure.train_crop_h,
                                                                           configure.train_crop_w),
                                                                     scale=(configure.train_resize_area_ratio_min,
                                                                            configure.train_resize_area_ratio_max),
                                                                     ratio=(configure.train_crop_ratio,
                                                                            configure.train_crop_ratio)))
        elif configure.train_crop_type == 'center_crop':
            train_transform_list.append(transforms.Resize(configure.train_resize_short_edge))
            train_transform_list.append(transforms.CenterCrop((configure.train_crop_h, configure.train_crop_w)))
        else:
            raise RuntimeError

        if configure.data_advance[0] == 'none':
            pass
        elif configure.data_advance[0] == 'color_diff_121_abs_3ch':
            train_transform_list.append(data_argumentation.ColorDiff121abs3ch())
        elif configure.data_advance[0] == 'color_diff_121_abs_1ch':
            train_transform_list.append(data_argumentation.ColorDiff121abs1ch())
        else:
            raise RuntimeError
    else:
        raise RuntimeError

    if configure.train_random_horizontal_flip:
        train_transform_list.append(transforms.RandomHorizontalFlip())
    if not configure.data_sampler:
        train_transform_list.append(normalize)

    # 設定 val_transform_list
    if configure.rgb_swap_order is not None:
        val_transform_list.append(data_argumentation.ChannelSwap(configure.rgb_swap_order))
    if configure.data_advance_num > 1:
        if 'color_diff_121_abs_3ch' in configure.data_advance or 'color_diff_121_abs_1ch' in configure.data_advance:
            val_transform_list.append(transforms.RandomResizedCrop(size=(configure.val_crop_h + 2,
                                                                         configure.val_crop_w + 2),
                                                                   scale=(configure.val_resize_area_ratio_min,
                                                                          configure.val_resize_area_ratio_max),
                                                                   ratio=(configure.val_crop_ratio,
                                                                          configure.val_crop_ratio)))
            val_transform_list.append(data_argumentation.AppendDataAdvance())
        else:
            raise RuntimeError
    elif configure.data_advance_num == 1:
        # 若有使用 conv2D 則 crop 時要留 padding 的空間
        if 'color_diff' in configure.data_advance[0]:
            configure.val_crop_h += 2
            configure.val_crop_w += 2

        if configure.val_crop_type == 'random_crop':
            val_transform_list.append(transforms.RandomResizedCrop(size=(configure.val_crop_h, configure.val_crop_w),
                                                                   scale=(configure.val_resize_area_ratio_min,
                                                                          configure.val_resize_area_ratio_max),
                                                                   ratio=(configure.val_crop_ratio,
                                                                          configure.val_crop_ratio)))
        elif configure.val_crop_type == 'center_crop':
            val_transform_list.append(transforms.Resize(configure.val_resize_short_edge))
            val_transform_list.append(transforms.CenterCrop((configure.val_crop_h, configure.val_crop_w)))
        else:
            raise RuntimeError

        if configure.data_advance[0] == 'none':
            pass
        elif configure.data_advance[0] == 'color_diff_121_abs_3ch':
            val_transform_list.append(data_argumentation.ColorDiff121abs3ch())
        elif configure.data_advance[0] == 'color_diff_121_abs_1ch':
            val_transform_list.append(data_argumentation.ColorDiff121abs1ch())
        else:
            raise RuntimeError
    else:
        raise RuntimeError

    if not configure.data_sampler:
        val_transform_list.append(normalize)

    train_dataset = datasets.ImageFolder(train_dir, transforms.Compose(train_transform_list))
    val_dataset = datasets.ImageFolder(val_dir, transforms.Compose(val_transform_list))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=configure.train_batch_size,
        shuffle=configure.train_shuffle,
        num_workers=configure.data_loader_worker,
        pin_memory=True,
        sampler=None
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=configure.val_batch_size,
        shuffle=False,
        num_workers=configure.data_loader_worker,
        pin_memory=True
    )

    if configure.data_sampler:
        if configure.sampled_loader == 'train':
            sampled_loader = train_loader
        elif configure.sampled_loader == 'val':
            sampled_loader = val_loader
        else:
            raise RuntimeError

        # visualizing data loader
        batch_count = 0
        for batch_instance in iter(sampled_loader):
            instance_count = 0
            for instance_No in range(configure.train_batch_size):
                (image_tensor, label_tensor) = (batch_instance[0][instance_No], batch_instance[1][instance_No])
                channel_first_image = np.array(image_tensor, dtype=float)
                if configure.input_channel >= 3:
                    channel_r = channel_first_image[0, ..., np.newaxis] * 255
                    channel_g = channel_first_image[1, ..., np.newaxis] * 255
                    channel_b = channel_first_image[2, ..., np.newaxis] * 255
                elif configure.input_channel == 1:
                    channel_r = channel_first_image[0, ..., np.newaxis] * 255
                    channel_g = channel_first_image[0, ..., np.newaxis] * 255
                    channel_b = channel_first_image[0, ..., np.newaxis] * 255
                else:
                    raise RuntimeError
                image = np.concatenate((channel_r, channel_g, channel_b), axis=-1)
                image = np.array(image, dtype=np.uint8)
                label = label_tensor.item()
                print(
                    'batch_count={0:05d}'.format(batch_count),
                    'instance_count={0:05d}'.format(instance_count),
                    'label:{}'.format(label)
                )
                # image cv2 show
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.imshow('image', image)
                cv2.waitKey()
                instance_count += 1
            batch_count += 1

    if configure.evaluate_only:
        validate(val_loader, model, criterion, log_rec)
        return

    for epoch in range(configure.start_epoch, configure.epochs):
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, log_rec)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, log_rec)

        scheduler.step(acc1)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        save_checkpoint({
            'epoch': epoch,
            'arch': configure.model,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }, is_best)
        if best_acc1 == best_acc1_early_stop:
            count_early_stop += 1
        else:
            best_acc1_early_stop = best_acc1
            count_early_stop = 0
        if count_early_stop == configure.early_stop:
            break


def train(train_loader, model, criterion, optimizer, epoch, log_rec):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('accuracy', ':.2f')
    # top1 = AverageMeter('Acc@1', ':6.2f')
    # top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [
            # batch_time,
            # data_time,
            losses,
            top1,
            # top5
        ],
        # prefix="Epoch {}".format(epoch)
    )

    # switch to train mode
    model.train()

    end = time.time()
    tqdm_control = trange(len(train_loader), desc='Epoch {} train: '.format(epoch), leave=True, ascii='->>',
                          bar_format='{desc}{n}/{total}[{bar:30}]{percentage:3.0f}% - {elapsed}{postfix}')
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if configure.specified_GPU_ID is not None:
            images = images.cuda(configure.specified_GPU_ID, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(configure.specified_GPU_ID, non_blocking=True)

        # compute output
        if configure.ina_type is None:
            predict = model(images)
            loss = criterion['CrossEntropy'](predict, target)
        else:
            predict, feature_pair_list = model(images)
            loss = criterion['CrossEntropy'](predict, target) * configure.ina_loss_weight[0]
            for loss_pair_No in range(configure.loss_pair_num):
                loss += criterion[configure.loss_function](feature_pair_list[loss_pair_No][0], feature_pair_list[loss_pair_No][1])\
                        * configure.ina_loss_weight[loss_pair_No + 1]

        # measure accuracy and record loss
        acc1, acc5 = accuracy(predict, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        # top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # if i % args.print_freq == 0:
        #     progress.display(i)
        if i == len(train_loader) - 1:
            log_rec.write('Epoch {} train: '.format(epoch) + progress.display(i) + '%')
        tqdm_control.set_postfix_str(progress.display(i) + '%')
        tqdm_control.update(1)
        tqdm_control.refresh()
    # progress.display(len(train_loader))
    del tqdm_control


def validate(val_loader, model, criterion, log_rec):
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('accuracy', ':.2f', Summary.AVERAGE)
    # top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    # top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader),
        [
            # batch_time,
            losses,
            top1,
            # top5
        ],
        # prefix='Test: '
    )

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        tqdm_control = trange(len(val_loader), desc='\t\tvalidation: ', leave=True, ascii='->>',
                              bar_format='{desc}{n}/{total}[{bar:30}]{percentage:3.0f}% - {elapsed}{postfix}')
        for i, (images, target) in enumerate(val_loader):
            if configure.specified_GPU_ID is not None:
                images = images.cuda(configure.specified_GPU_ID, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(configure.specified_GPU_ID, non_blocking=True)

            # compute output
            if configure.ina_type is None:
                predict = model(images)
                loss = criterion['CrossEntropy'](predict, target)
            else:
                predict, feature_pair_list = model(images)
                loss = criterion['CrossEntropy'](predict, target) * configure.ina_loss_weight[0]
                for loss_pair_No in range(configure.loss_pair_num):
                    loss += criterion[configure.loss_function](feature_pair_list[loss_pair_No][0], feature_pair_list[loss_pair_No][1]) \
                            * configure.ina_loss_weight[loss_pair_No + 1]

            # measure accuracy and record loss
            acc1, acc5 = accuracy(predict, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            # top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # if i % args.print_freq == 0:
            #     progress.display(i)
        # progress.display(len(val_loader))
        # progress.display_summary()
            if i == len(val_loader) - 1 and not configure.evaluate_only:
                log_rec.write('\t\tvalidation: ' + progress.display(i) + '%')
            tqdm_control.set_postfix_str(progress.display(i) + '%')
            tqdm_control.update(1)
            tqdm_control.refresh()
        del tqdm_control

    return top1.avg


def save_checkpoint(state, is_best):
    filename = 'checkpoint.pth.tar'
    torch.save(state, configure.ckpt_dir + filename)
    if is_best:
        shutil.copyfile(configure.ckpt_dir + filename, configure.ckpt_dir + 'model_best.pth.tar')


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        # fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        fmtstr = '{name}: {avg' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        # entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries = self.prefix
        for meter in self.meters:
            entries += ' - ' + str(meter)
        return str(entries)

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        return str(entries)

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
