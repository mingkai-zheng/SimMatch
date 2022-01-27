# some code in this file is adapted from
# https://github.com/pytorch/examples
# Original Copyright 2017. Licensed under the BSD 3-Clause License.
# Modifications Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import argparse
import builtins
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from data.imagenet import *

import backbone as backbone_models
from models.simmatch import get_simmatch_model
from utils import utils, lr_schedule, get_norm, dist_utils
import data.transforms as data_transforms
from engine import validate


backbone_model_names = sorted(name for name in backbone_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(backbone_models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--trainindex_x', default=None, type=str, metavar='PATH',
                    help='path to train annotation_x (default: None)')
parser.add_argument('--trainindex_u', default=None, type=str, metavar='PATH',
                    help='path to train annotation_u (default: None)')
parser.add_argument('--arch', metavar='ARCH', default='SimMatch',
                    help='model architecture')
parser.add_argument('--backbone', default='resnet50_encoder',
                    choices=backbone_model_names,
                    help='model architecture: ' +
                        ' | '.join(backbone_model_names) +
                        ' (default: resnet50_encoder)')
parser.add_argument('--cls', default=1000, type=int, metavar='N',
                    help='number of classes')
parser.add_argument('--port', default=23456, type=int, help='dist init port')                    
parser.add_argument('-j', '--workers', default=6, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--warmup-epoch', default=0, type=int, metavar='N',
                    help='number of epochs for learning warmup')
parser.add_argument('-b', '--batch-size', default=8, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--nesterov', action='store_true', default=False,
                    help='use nesterov momentum')
parser.add_argument('-p', '--print-freq', default=20, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--eval-freq', default=1, type=int,
                    metavar='N', help='evaluation epoch frequency (default: 1)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', default='', type=str, metavar='PATH',
                    help='path to pretrained model (default: none)')
parser.add_argument('--self-pretrained', default='', type=str, metavar='PATH',
                    help='path to MoCo pretrained model (default: none)')
parser.add_argument('--super-pretrained', default='', type=str, metavar='PATH',
                    help='path to supervised pretrained model (default: none)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

parser.add_argument('--anno-percent', type=float, default=0.1,
                    help='number of labeled data')
parser.add_argument('--split-seed', default=42, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--mu', default=5, type=int,
                    help='coefficient of unlabeled batch size')
parser.add_argument('--lambda-u', default=10, type=float,
                    help='coefficient of unlabeled loss')
parser.add_argument('--threshold', default=0.7, type=float,
                    help='pseudo label threshold')
parser.add_argument('--eman', action='store_true', default=False,
                    help='use EMAN')
parser.add_argument('--ema-m', default=0.999, type=float,
                    help='EMA decay rate')
parser.add_argument('--norm', default='None', type=str,
                    help='the normalization for backbone (default: None)')
# online_net.backbone for BYOL
parser.add_argument('--moco-path', default=None, type=str)
parser.add_argument('--model-prefix', default='encoder_q', type=str,
                    help='the model prefix of self-supervised pretrained state_dict')
parser.add_argument('--st', type=float, default=0.1)
parser.add_argument('--tt', type=float, default=0.1)
parser.add_argument('--c_smooth', type=float, default=1.0)
parser.add_argument('--DA', default=False, action='store_true')
parser.add_argument('--lambda_in', type=float, default=1)
parser.add_argument('--randaug', default=False, action='store_true')
parser.add_argument('--stack', default=False, action='store_true')
args = parser.parse_args()


def main_worker():
    best_acc1 = 0
    best_acc5 = 0

    rank, local_rank, world_size = dist_utils.dist_init(args.port)
    args.gpu = local_rank
    args.rank = rank
    args.world_size = world_size
    args.distributed = True
    

    if rank != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass
    
    print(args)
    
    train_dataset_x, train_dataset_u, val_dataset = get_imagenet_ssl()

    # Data loading code
    train_sampler = DistributedSampler

    train_loader_x = DataLoader(
        train_dataset_x,
        sampler=train_sampler(train_dataset_x),
        batch_size=args.batch_size, persistent_workers=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    train_loader_u = DataLoader(
        train_dataset_u,
        sampler=train_sampler(train_dataset_u),
        batch_size=args.batch_size * args.mu, persistent_workers=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    val_loader = DataLoader(
        val_dataset,
        sampler=train_sampler(val_dataset),
        batch_size=64, shuffle=False, persistent_workers=True,
        num_workers=args.workers, pin_memory=True)
    


    # create model
    print("=> creating model '{}' with backbone '{}'".format(args.arch, args.backbone))
    model_func = get_simmatch_model(args.arch)
    norm = get_norm(args.norm)
    model = model_func(
        backbone_models.__dict__[args.backbone],
        eman=args.eman,
        momentum=args.ema_m,
        norm=norm,
        K=len(train_dataset_x),
        args=args
    )
    
    if args.moco_path is not None:
        checkpoint = torch.load(args.moco_path, map_location="cpu")
        state_dict = checkpoint['state_dict']
        for k in list(state_dict.keys()):
            if k.startswith('module.encoder_q'):
                state_dict[k.replace('module.encoder_q', 'backbone')] = state_dict[k]
            del state_dict[k]
        
        for k in list(state_dict.keys()):
            if 'backbone.fc.0' in k:
                state_dict[k.replace('backbone.fc.0','head.0')] = state_dict[k]
                del state_dict[k]
            if 'backbone.fc.2' in k:
                state_dict[k.replace('backbone.fc.2','head.2')] = state_dict[k]            
                del state_dict[k]
        
        model.main.load_state_dict(state_dict=state_dict, strict=False)

        for param, param_m in zip(model.main.parameters(), model.ema.parameters()):
            param_m.data.copy_(param.data)  
            param_m.requires_grad = False


    model.cuda(args.gpu)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)

    # define loss function (criterion)
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=args.nesterov)

    checkpoint_path = 'checkpoints/{}'.format(args.checkpoint)
    print('checkpoint_path:', checkpoint_path)
    if os.path.exists(checkpoint_path):
        checkpoint =  torch.load(checkpoint_path, map_location='cpu')
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})".format(checkpoint_path, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(checkpoint_path))

    cudnn.benchmark = True

    if args.evaluate:
        acc1, acc5 = validate(val_loader, model, criterion, args)
        if rank == 0:
            print('* Acc@1 {:.3f} Acc@5 {:.3f}'.format(acc1, acc5))

    else:
        for epoch in range(args.start_epoch, args.epochs):
            if epoch >= args.warmup_epoch:
                lr_schedule.adjust_learning_rate(optimizer, epoch, args)

            # train for one epoch
            train(train_loader_x, train_loader_u, model, optimizer, epoch, args)

            if (epoch + 1) % args.eval_freq == 0:
                # evaluate on validation set
                acc1, acc5 = validate(val_loader, model, criterion, args)
                # remember best acc@1 and save checkpoint
                best_acc1 = max(acc1, best_acc1)
                best_acc5 = max(acc5, best_acc5)

            if rank == 0:
                print('Epoch:{} * Acc@1 {:.3f} Acc@5 {:.3f} Best_Acc@1 {:.3f} Best_Acc@5 {:.3f}'.format(epoch, acc1, acc5, best_acc1, best_acc5))
                torch.save({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, checkpoint_path)


def train(train_loader_x, train_loader_u, model, optimizer, epoch, args):
    batch_time = utils.AverageMeter('Time', ':6.3f')
    data_time = utils.AverageMeter('Data', ':6.3f')
    losses = utils.AverageMeter('Loss', ':.4e')
    losses_x = utils.AverageMeter('Loss_x', ':.4e')
    losses_u = utils.AverageMeter('Loss_u', ':.4e')
    losses_in = utils.AverageMeter('Loss_in', ':.4e')
    top1_x = utils.AverageMeter('Acc_x@1', ':6.2f')
    top5_x = utils.AverageMeter('Acc_x@5', ':6.2f')
    top1_u = utils.AverageMeter('Acc_u@1', ':6.2f')
    top5_u = utils.AverageMeter('Acc_u@5', ':6.2f')
    mask_info = utils.AverageMeter('Mask', ':6.3f')
    pseudo_label_info = utils.AverageMeter('Label', ':6.3f')
    curr_lr = utils.InstantMeter('LR', '')
    progress = utils.ProgressMeter(
        len(train_loader_u),
        [curr_lr, batch_time, data_time, losses, losses_x, losses_u, losses_in, top1_x, top5_x, top1_u, top5_u, mask_info, pseudo_label_info],
        prefix="Epoch: [{}/{}]\t".format(epoch, args.epochs))

    epoch_x = epoch * math.ceil(len(train_loader_u) / len(train_loader_x))
    if args.distributed:
        print("set epoch={} for labeled sampler".format(epoch_x))
        train_loader_x.sampler.set_epoch(epoch_x)
        print("set epoch={} for unlabeled sampler".format(epoch))
        train_loader_u.sampler.set_epoch(epoch)

    train_iter_x = iter(train_loader_x)
    # switch to train mode
    model.train()
    if args.eman:
        print("setting the ema model to eval mode")
        if hasattr(model, 'module'):
            model.module.ema.eval()
        else:
            model.ema.eval()

    end = time.time()
    for i, (images_u, targets_u) in enumerate(train_loader_u):
        try:
            images_x, targets_x, index = next(train_iter_x)
        except Exception:
            epoch_x += 1
            print("reshuffle train_loader_x at epoch={}".format(epoch_x))
            if args.distributed:
                print("set epoch={} for labeled sampler".format(epoch_x))
                train_loader_x.sampler.set_epoch(epoch_x)
            train_iter_x = iter(train_loader_x)
            images_x, targets_x, index = next(train_iter_x)

        images_u_w, images_u_s = images_u
        # measure data loading time
        data_time.update(time.time() - end)

    
        images_x = images_x.cuda(args.gpu, non_blocking=True)
        images_u_w = images_u_w.cuda(args.gpu, non_blocking=True)
        images_u_s = images_u_s.cuda(args.gpu, non_blocking=True)
        targets_x = targets_x.cuda(args.gpu, non_blocking=True)
        targets_u = targets_u.cuda(args.gpu, non_blocking=True)
        index = index.cuda(args.gpu, non_blocking=True)


        # warmup learning rate
        if epoch < args.warmup_epoch:
            warmup_step = args.warmup_epoch * len(train_loader_u)
            curr_step = epoch * len(train_loader_u) + i + 1
            lr_schedule.warmup_learning_rate(optimizer, curr_step, warmup_step, args)
        curr_lr.update(optimizer.param_groups[0]['lr'])

        # model forward
        logits_x, pseudo_label, logits_u_s, loss_in = model(images_x, images_u_w, images_u_s, labels=targets_x, index=index, start_unlabel=epoch>0, args=args)
        max_probs, _ = torch.max(pseudo_label, dim=-1)
        mask = max_probs.ge(args.threshold).float()

        loss_x = F.cross_entropy(logits_x, targets_x, reduction='mean')
        loss_u = (torch.sum(-F.log_softmax(logits_u_s,dim=1) * pseudo_label.detach(), dim=1) * mask).mean()
        
        loss_in = loss_in.mean()
        loss = loss_x + args.lambda_u * loss_u + args.lambda_in * loss_in

        # measure accuracy and record loss
        losses.update(loss.item())
        losses_x.update(loss_x.item(), images_x.size(0))
        losses_u.update(loss_u.item(), images_u_w.size(0))
        losses_in.update(loss_in.item(), images_u_w.size(0))
        acc1_x, acc5_x = utils.accuracy(logits_x, targets_x, topk=(1, 5))
        top1_x.update(acc1_x[0], logits_x.size(0))
        top5_x.update(acc5_x[0], logits_x.size(0))
        acc1_u, acc5_u = utils.accuracy(pseudo_label, targets_u, topk=(1, 5))
        top1_u.update(acc1_u[0], pseudo_label.size(0))
        top5_u.update(acc5_u[0], pseudo_label.size(0))
        mask_info.update(mask.mean().item(), mask.size(0))

        bool_mask = mask.bool()
        psudo_label_correct = sum(pseudo_label.max(1)[1][bool_mask] == targets_u[bool_mask]) / (bool_mask.sum() + 1e-8)
        pseudo_label_info.update(psudo_label_correct * 100)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # update the ema model
        if hasattr(model, 'module'):
            model.module.momentum_update_ema()
        else:
            model.momentum_update_ema()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def get_imagenet_ssl(val_type='DefaultVal'):
    transform_x = data_transforms.weak_aug
    weak_transform = data_transforms.weak_aug
    if args.stack:
        strong_transform = data_transforms.stack_aug
    if args.randaug:
        strong_transform = data_transforms.rand_aug
    else:
        strong_transform = data_transforms.moco_aug
    transform_u = data_transforms.TwoCropsTransform(weak_transform, strong_transform)
    transform_val = data_transforms.get_transforms(val_type)

    train_dataset_x = ImagenetPercentV2(percent=args.anno_percent, labeled=True, aug=transform_x, return_index=True)
    train_dataset_u = ImagenetPercentV2(percent=args.anno_percent, labeled=False, aug=transform_u)
    val_dataset = Imagenet(mode='val', aug=transform_val)

    return train_dataset_x, train_dataset_u, val_dataset



if __name__ == '__main__':
    main_worker()
