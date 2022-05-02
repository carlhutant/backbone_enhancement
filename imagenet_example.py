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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from tqdm import trange

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch Backbone Training')
parser.add_argument('--data', metavar='DIR', default=configure.data_dir,
                    help='path to dataset (default: imagenet)')
parser.add_argument('-a', '--arch', metavar='ARCH', default=configure.model,
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=configure.data_loader_worker, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=configure.epochs, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=configure.train_batch_size, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=configure.lr, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=configure.momentum, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=configure.weight_decay, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('-e', '--evaluate', default=configure.evaluate_only, dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=configure.random_seed, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=configure.specified_GPU_ID, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

best_acc1 = 0
best_acc1_early_stop = 0
count_early_stop = 0


def main():
    # 保存 configure
    if not configure.evaluate_only:
        log_rec = log_record.LogRecoder(configure.resume)
        if os.path.exists(Path(configure.ckpt_dir).joinpath('configure.py')):
            raise RuntimeError
        else:
            shutil.copyfile(Path(configure.code_dir).joinpath('configure.py'),
                            Path(configure.ckpt_dir).joinpath('configure.py'))
    else:
        log_rec = None
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        if configure.swap_evaluate:
            origin_list = [[0, 1, 2],
                           [0, 2, 1],
                           [1, 0, 2],
                           [1, 2, 0],
                           [2, 0, 1],
                           [2, 1, 0]]
            for origin in origin_list:
                head = 0
                order = []
                for model_No in range(configure.model_num):
                    if '1ch' in configure.data_advance[model_No]:
                        order.append(head)
                        head += 1
                    else:
                        order += [x + head for x in origin]
                        head += 3
                configure.rgb_swap_order = order
                print('order:{}'.format(configure.rgb_swap_order))
                main_worker(args.gpu, ngpus_per_node, log_rec, args)
        else:
            # Simply call main_worker function
            main_worker(args.gpu, ngpus_per_node, log_rec, args)


def main_worker(gpu, ngpus_per_node, log_rec, args):
    global best_acc1
    global best_acc1_early_stop
    global count_early_stop
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    if configure.model_num < 1:
        raise RuntimeError
    elif configure.model_num > 1:
        model = concat_backbone.ConcatResNet()
    else:
        if 'resnet' in args.arch[0]:
            if 'fix_backbone' in configure.model_mode[0]:
                model = finetune.FineTuneResNet(model_id=0)
            elif args.arch[0] == 'resnet50':
                model = ResNet.resnet50(model_id=0)
            elif args.arch[0] == 'resnet101':
                model = ResNet.resnet101(model_id=0)
            else:
                raise RuntimeError
        else:
            raise RuntimeError
            # if args.pretrained:
            #     print("=> using pre-trained model '{}'".format(args.arch))
            #     model = models.__dict__[args.arch](pretrained=True)
            # else:
            #     print("=> creating model '{}'".format(args.arch))
            #     model = models.__dict__[args.arch]()

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs of the current node.
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # print(model)
    # for name, param in model.named_parameters():
    #     a = 0

    # define loss function (criterion), optimizer, and learning rate scheduler
    criterion = {'CrossEntropy': nn.CrossEntropyLoss().cuda(args.gpu),
                 'MSE': nn.MSELoss().cuda(args.gpu)}

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # LR scheduler
    # ReduceLROnPlateau
    scheduler = ReduceLROnPlateau(optimizer, mode='max',
                                  factor=configure.factor,
                                  patience=configure.patience,
                                  threshold=configure.threshold,
                                  verbose=True)
    # # Sets the learning rate to the initial LR decayed by 10 every 30 epochs
    # scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    # resume from a checkpoint
    if configure.resume:
        if configure.model_num > 1:  # 使用 concat backbone
            model.load(args)
        elif configure.model_num == 1 and 'fix' in configure.model_mode[0]:  # 使用 finetune
            model.load(args)
        else:  # 使用 ResNet
            if os.path.isfile(configure.resume_ckpt_path[0]):
                print("=> loading checkpoint '{}'".format(configure.resume_ckpt_path[0]))
                if args.gpu is None:
                    checkpoint = torch.load(configure.resume_ckpt_path[0])
                else:
                    # Map model to be loaded to specified single gpu.
                    loc = 'cuda:{}'.format(args.gpu)
                    checkpoint = torch.load(configure.resume_ckpt_path[0], map_location=loc)
                args.start_epoch = checkpoint['epoch']
                best_acc1 = checkpoint['best_acc1']
                if args.gpu is not None:
                    # best_acc1 may be from a checkpoint from a different GPU
                    best_acc1 = best_acc1.to(args.gpu)
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
    train_dir = os.path.join(args.data, 'train')
    val_dir = os.path.join(args.data, 'val')
    # imagenet rgb mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    mean = []
    std = []
    for data_advance in configure.data_advance:
        if configure.dataset == 'AWA2':
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
        elif configure.dataset == 'inat2021':
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
        elif configure.dataset == 'office-31':
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
        else:
            raise RuntimeError
    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform_list = [transforms.ToTensor()]
    val_transform_list = [transforms.ToTensor()]

    # 設定 train_transform_list
    if configure.data_advance_num > 1:
        if 'color_diff_121_abs_3ch' in configure.data_advance or 'color_diff_121_abs_1ch' in configure.data_advance:
            train_transform_list.append(transforms.RandomResizedCrop(size=(configure.train_crop_h + 2,
                                                                           configure.train_crop_w + 2),
                                                                     scale=(configure.train_resize_area_ratio_min,
                                                                            configure.train_resize_area_ratio_max),
                                                                     ratio=(configure.train_crop_ratio,
                                                                            configure.train_crop_ratio)))
            train_transform_list.append(data_argumentation.AppendDataAdvance())
        # elif configure.data_advance1 == 'none' and configure.data_advance2 == 'color_diff_121_abs_1ch':
        #     train_compose_list.append(transforms.RandomResizedCrop(size=(configure.train_crop_h + 2,
        #                                                                  configure.train_crop_w + 2),
        #                                                            scale=(configure.train_resize_area_ratio_min,
        #                                                                   configure.train_resize_area_ratio_max),
        #                                                            ratio=(configure.train_crop_ratio,
        #                                                                   configure.train_crop_ratio)))
        #     train_compose_list.append(data_argumentation.AppendColorDiff121abs1ch())
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
    if configure.rgb_swap_order is not None:
        train_transform_list.append(data_argumentation.ChannelSwap(configure.rgb_swap_order))
    if not configure.data_sampler:
        train_transform_list.append(normalize)

    # 設定 val_transform_list
    if configure.data_advance_num > 1:
        if 'color_diff_121_abs_3ch' in configure.data_advance or 'color_diff_121_abs_1ch' in configure.data_advance:
            val_transform_list.append(transforms.RandomResizedCrop(size=(configure.val_crop_h + 2,
                                                                         configure.val_crop_w + 2),
                                                                   scale=(configure.val_resize_area_ratio_min,
                                                                          configure.val_resize_area_ratio_max),
                                                                   ratio=(configure.val_crop_ratio,
                                                                          configure.val_crop_ratio)))
            val_transform_list.append(data_argumentation.AppendDataAdvance())
        # elif configure.data_advance1 == 'none' and configure.data_advance2 == 'color_diff_121_abs_1ch':
        #     val_transform_list.append(transforms.RandomResizedCrop(size=(configure.val_crop_h + 2,
        #                                                                configure.val_crop_w + 2),
        #                                                          scale=(configure.val_resize_area_ratio_min,
        #                                                                 configure.val_resize_area_ratio_max),
        #                                                          ratio=(configure.val_crop_ratio,
        #                                                                 configure.val_crop_ratio)))
        #     val_transform_list.append(data_argumentation.AppendColorDiff121abs1ch())
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

    if configure.rgb_swap_order is not None:
        val_transform_list.append(data_argumentation.ChannelSwap(configure.rgb_swap_order))
    if not configure.data_sampler:
        val_transform_list.append(normalize)

    train_dataset = datasets.ImageFolder(train_dir, transforms.Compose(train_transform_list))
    val_dataset = datasets.ImageFolder(val_dir, transforms.Compose(val_transform_list))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None
    # train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
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
            for instance_No in range(args.batch_size):
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

    if args.evaluate:
        validate(val_loader, model, criterion, log_rec, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, log_rec, args)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, log_rec, args)

        scheduler.step(acc1)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }, is_best, '{:03d}'.format(epoch + 1))
        if best_acc1 == best_acc1_early_stop:
            count_early_stop += 1
        else:
            best_acc1_early_stop = best_acc1
            count_early_stop = 0
        if count_early_stop == configure.early_stop:
            break


def train(train_loader, model, criterion, optimizer, epoch, log_rec, args):
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
    tqdm_control = trange(len(train_loader), desc='Epoch {} train: '.format(epoch + 1), leave=True, ascii='->>',
                          bar_format='{desc}{n}/{total}[{bar:30}]{percentage:3.0f}% - {elapsed}{postfix}')
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)

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
            log_rec.write('Epoch {} train: '.format(epoch + 1) + progress.display(i) + '%')
        tqdm_control.set_postfix_str(progress.display(i) + '%')
        tqdm_control.update(1)
        tqdm_control.refresh()
    # progress.display(len(train_loader))
    del tqdm_control


def validate(val_loader, model, criterion, log_rec, args):
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
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

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


def save_checkpoint(state, is_best, epoch):
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
