import argparse
import math
import shutil
import time
import warnings
import os

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.distributed as dist
import torch.utils.data
import torchvision.models as torchvision_models

import models
from dataloader.spatio_temporal_dataset import SpatioTemporalDataset
from models.spatial_stream import SpatialStream
from models.temporal_stream import TemporalStream
from models.two_stream_fusion import TwoStreamFusion

try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this code.")

"""
Based on:
https://github.com/pytorch/examples/blob/master/imagenet/main.py
and
https://github.com/NVIDIA/apex/blob/master/examples/imagenet/main_amp.py
"""


def get_model_names(models):
    return [name for name in models.__dict__ if name.islower() and not name.startswith('__')
            and callable(models.__dict__[name])]


model_names = sorted(get_model_names(models) + get_model_names(torchvision_models))

train_modes = ['spatio_temporal', 'spatial', 'temporal']

parser = argparse.ArgumentParser(description='UCF101 Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--data_split', default='01',
                    help='Data split csv file')
parser.add_argument('-a', '--arch', metavar='ARCH', default='vgg16_bn', choices=model_names,
                    help='backbone architecture: ' + ' | '.join(model_names) + ' (default: vgg16_bn)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=120, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts')
parser.add_argument('-b', '--batch-size', default=8, type=int, metavar='N',
                    help='mini-batch size (default:64) is the total batch size of all GPUs on the'
                         'current node when using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, metavar='LR',
                    help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight decay', default=1e-4, type=float, metavar='W',
                    help='weight decay (default:1e-4)', dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int, metavar='N',
                    help='print frequency (default:10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to the latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model', default=True)
parser.add_argument('--mode', default='spatio_temporal', type=str,
                    help='Train two-stream network fusion or only one stream.')

parser.add_argument("--local_rank", default=0, type=int)

parser.add_argument('--opt-level', default='O1', type=str)
parser.add_argument('--keep-batchnorm-fp32', type=str, default=None)
parser.add_argument('--loss-scale', type=str, default=None)

best_acc1 = 0

def main():
    args = parser.parse_args()

    # benchmark mode will look for the optimal set of algorithms for a particular configuration
    # it might lead to faster runtime unless the input size changes at each iteration
    cudnn.benchmark = True

    if args.local_rank == 0:
        print('\nCUDNN VERSION: {}\n'.format(torch.backends.cudnn.version()))

    main_worker(args)


def main_worker(args):
    global best_acc1

    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1

    # if args.gpu is not None:
    #     print("User GPU: {} for training".format(args.gpu))
    args.gpu = 0
    args.world_size = 1

    if args.distributed:
        args.gpu = args.local_rank
        torch.cuda.set_device(args.gpu)
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')
        args.world_size = torch.distributed.get_world_size()

    assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."

    model_zoo = models if args.arch in models.__dict__ else torchvision_models
    if args.pretrained:
        if args.local_rank == 0:
            print("=> using pre-trained model '{}'".format(args.arch))
        model = model_zoo.__dict__[args.arch](pretrained=True)
    else:
        if args.local_rank == 0:
            print("=> creating model '{}'".format(args.arch))
        model = model_zoo.__dict__[args.arch]()

    if args.mode == 'spatial':
        spatial_stream = SpatialStream(model, args.arch, num_classes=101)
        model = spatial_stream
    elif args.mode == 'temporal':
        temporal_stream = TemporalStream(model, args.arch, num_classes=101)
        model = temporal_stream
    elif args.mode == 'spatio_temporal':
        two_stream_fusion = TwoStreamFusion(101, model, args.arch)
        model = two_stream_fusion

    print(model)
    model.cuda()

    # Scale learning rate based on global batch size
    args.lr = args.lr * float(args.batch_size * args.world_size) / 256.

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # Initialize Amp.  Amp accepts either values or strings for the optional override arguments,
    # for convenient inter-operation with argparse.
    model, optimizer = amp.initialize(model, optimizer,
                                      opt_level=args.opt_level,
                                      keep_batchnorm_fp32=args.keep_batchnorm_fp32,
                                      loss_scale=args.loss_scale
                                      )

    if args.distributed:
        model = DDP(model, delay_allreduce=True)

    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    # Optionally resume from a checkpoint
    if args.resume:
        # Use a local scope to avoid dangling references
        def resume():
            if os.path.isfile(args.resume):
                print("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda(args.gpu))
                args.start_epoch = checkpoint['epoch']
                best_acc1 = checkpoint['best_acc1']
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.resume, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))
        resume()

    train_dataset = SpatioTemporalDataset(args.data, 'trainlist{}.csv'.format(args.data_split))

    val_dataset = SpatioTemporalDataset(args.data, 'vallist{}.csv'.format(args.data_split))

    train_sampler = None
    val_sampler = None
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    # if args.benchmark:
    #
    #     validate(val_loader, model, criterion, args)
    #     return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        # train(spatial_train_loader, model, criterion, optimizer, epoch, args)
        train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict()
        }, is_best, args.arch)


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(train_loader), batch_time, data_time, losses, top1,
                             top5, prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (spatial, temporal, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        adjust_learning_rate(optimizer, epoch, i, len(train_loader), args)

        if args.gpu is not None and (args.mode == 'spatial' or args.mode == 'spatio_temporal'):
            spatial = spatial.cuda(non_blocking=True)
        if args.gpu is not None and (args.mode == 'temporal' or args.mode == 'spatio_temporal'):
            temporal = temporal.cuda(non_blocking=True)
        target = target.cuda()

        # compute output
        if args.mode == 'spatial':
            output = model(spatial)
        elif args.mode == 'temporal':
            output = model(temporal)
        else:
            output = model(spatial, temporal)

        loss = criterion(output, target)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output.data, target, topk=(1, 5))

        if args.distributed:
            reduced_loss = reduce_tensor(loss.data, args)
            acc1 = reduce_tensor(acc1, args)
            acc5 = reduce_tensor(acc5, args)
        else:
            reduced_loss = loss.data

        losses.update(to_python_float(reduced_loss), spatial.size(0))
        top1.update(to_python_float(acc1), spatial.size(0))
        top5.update(to_python_float(acc5), spatial.size(0))

        torch.cuda.synchronize()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and args.local_rank == 0:
            progress.print(i)


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader), batch_time, losses, top1, top5, prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (spatial, temporal, target) in enumerate(val_loader):
            if args.gpu is not None and (args.mode == 'spatial' or args.mode == 'spatio_temporal'):
                spatial = spatial.cuda(non_blocking=True)
            if args.gpu is not None and (args.mode == 'temporal' or args.mode == 'spatio_temporal'):
                temporal = temporal.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            if args.mode == 'spatial':
                output = model(spatial)
            elif args.mode == 'temporal':
                output = model(temporal)
            else:
                output = model(spatial, temporal)

            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output.data, target, topk=(1, 5))

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args)
                acc1 = reduce_tensor(acc1, args)
                acc5 = reduce_tensor(acc5, args)
            else:
                reduced_loss = loss.data

            losses.update(to_python_float(reduced_loss), spatial.size(0))
            top1.update(to_python_float(acc1), spatial.size(0))
            top5.update(to_python_float(acc5), spatial.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0 and args.local_rank == 0:
                progress.print(i)

    return top1.avg


def save_checkpoint(state, is_best, arch):
    filename = 'checkpoint_{}.pth.tar'.format(arch)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best_{}.pth.tar'.format(arch))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def adjust_learning_rate(optimizer, epoch, step, len_epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    # if epoch < 5:
    #     lr = args.lr * float(1 + step + epoch * len_epoch) / (5. * len_epoch)
    # else:
    number_batches = args.epochs * len_epoch
    # current_batch = (epoch - 5) * len_epoch + step
    current_batch = epoch * len_epoch + step
    lr = 0.5 * (1 + math.cos(current_batch * math.pi / number_batches)) * args.lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


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
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def reduce_tensor(tensor, args):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= args.world_size
    return rt


if __name__ == '__main__':
    main()
