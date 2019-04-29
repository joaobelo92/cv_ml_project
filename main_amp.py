import argparse
import shutil
import time
import os
import math

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as torchvision_models

import numpy as np

import models
from dataloader.spatial_dataset import SpatialDataset
from models.spatial_stream import SpatialStream

try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError("Install apex from https://www.github.com/nvidia/apex")

"""
Training using mixed precision based on:
https://github.com/pytorch/examples/blob/master/imagenet/main.py
https://github.com/NVIDIA/apex/tree/master/examples/imagenet
"""


def get_model_names(models):
    return [name for name in models.__dict__ if name.islower() and not name.startswith('__')
            and callable(models.__dict__[name])]


model_names = sorted(get_model_names(models) + get_model_names(torchvision_models))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='Path to dataset')
parser.add_argument('data_split', metavar='DIR',
                    help='Filename of the data split csv file')
parser.add_argument('-a', '--arch', metavar='ARCH', default='vgg16_bn', choices=model_names,
                    help='Model architecture: ' + ' | '.join(model_names) + ' (default: vgg16_bn)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='Number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=120, type=int, metavar='N',
                    help='Number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='Manual epoch number (useful on restarts')
parser.add_argument('-b', '--batch_size', default=64, type=int, metavar='N',
                    help='Mini-batch size (default:64) is the total batch size of all GPUs on the'
                         'current node when using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning_rate', default=0.1, type=float, metavar='LR',
                    help='Initial learning rate. Rule of thumb - 0.1 for batch size 256', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='Momentum')
parser.add_argument('--wd', '--weight_decay', default=1e-4, type=float, metavar='W',
                    help='weight decay (default:1e-4)', dest='weight_decay')
parser.add_argument('-p', '--print_freq', default=50, type=int, metavar='N',
                    help='Print frequency (default:50)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to the latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', action='store_true',
                    help='Evaluate model on validation set')
parser.add_argument('--pretrained', default=True, dest='pretrained', action='store_true',
                    help='Use pre-trained model')

parser.add_argument('--local_rank', default=0, type=int,
                    help='This argument is used by torch.distributed')
parser.add_argument('--sync_bn', action='store_true',
                    help='Enabling apex sync BN.')

parser.add_argument('--opt_level', default='O1', type=str, choices=['O0', 'O1', 'O2', 'O3'],
                    help='Level of mixed precision: \n O0 - Pure FP32 training \n O1 - Conservative mixed precision \n'
                         'O2 - fast mixed precision \n O3 - Pure FP16 training')
parser.add_argument('--keep_batchnorm_fp32', type=str, default=None,
                    help='Allows PyTorch to use cudnn batchnorms. Only makes sense when doing pure FP16 training')
parser.add_argument('--loss_scale', type=str, default=None,
                    help='Small gradients may underflow in FP16. This issue is solved by multiplying the loss'
                         'by a constant S. By the chain rule, gradients will also be scaled by S and small gradients'
                         'will be preserved. Dynamic loss scaling is used by default.')

best_acc1 = 0


def fast_collate(batch):
    """Loads the batch into the gpu faster with some tricks"""
    images = [img[0] for img in batch]
    targets = torch.tensor([target[1] for target in batch])
    w = images[0].size[0]
    h = images[0].size[1]
    tensor = torch.zeros((len(images), 3, h, w), dtype=torch.uint8)
    for i, img in enumerate(images):
        np_array = np.asarray(img, dtype=np.uint8)
        tens = torch.from_numpy(np_array)
        if np_array.ndim < 3:
            np_array = np.expand_dims(np_array, axis=-1)
        np_array = np.rollaxis(np_array, 2)

        tensor[i] += torch.from_numpy(np_array)

    return tensor, targets


def main():
    args = parser.parse_args()
    main_worker(args)


def main_worker(args):
    end = time.time()
    global best_acc1

    assert cudnn.enabled, "Amp requires cudnn backend to be enabled"

    print("=> Optimization level: {}".format(args.opt_level))
    print("=> Compute batch normalization in FP32: {}".format(args.keep_batchnorm_fp32), type(args.keep_batchnorm_fp32))
    print("=> Loss scale: {}".format(args.loss_scale), type(args.loss_scale))
    print("=> CDNN version: {}".format(cudnn.version()))

    args.distributed = 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1
    args.world_size = 1
    gpu = 0

    if args.distributed:
        gpu = args.local_rank
        torch.cuda.set_device(gpu)
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')
        args.world_size = torch.distributed.get_world_size()

    model_zoo = models if args.arch in models.__dict__ else torchvision_models
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = model_zoo.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = model_zoo.__dict__[args.arch]()

    if args.sync_bn:
        import apex
        print("=> Using apex synced BN")
        model = apex.parallel.convert_syncbn_model(model)

    spatial_stream = SpatialStream(101, model, args.arch)

    model = spatial_stream.cuda()

    # Scale learning based on global batch size, similar to He's residual nets paper
    args.lr = args.lr * float(args.batch_size * args.world_size) / 256.
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=False)

    # Initialize Amp. Amp accepts either values or strings for the optional override arguments,
    # for convenient interoperation with argparse.
    model, optimizer = amp.initialize(model, optimizer,
                                      opt_level=args.opt_level,
                                      keep_batchnorm_fp32=args.keep_batchnorm_fp32,
                                      loss_scale=args.loss_scale)

    # For distributed training, wrap the model with apex.parallel.DistributedDataParallel.
    # This must be done AFTER the call to amp.initialize.
    if args.distributed:
        # By default, apex.parallel.DistributedParallel overlaps communication with computation in the backward pass.
        # delay_allreduce delays all communication to the end of the backward pass
        model = DDP(model, delay_allreduce=True)

    # TODO: Consider label smoothing
    # Define loss function (criterion)
    criterion = nn.CrossEntropyLoss().cuda()  # .cuda(args.gpu)

    # Optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> Loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda(gpu))
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> Loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> No checkpoint found at '{}'".format(args.resume))

    # benchmark mode will look for the optimal set of algorithms for a particular configuration
    # it might lead to faster runtime unless the input size changes at each iteration
    cudnn.benchmark = True

    train_dir = os.path.join(args.data, 'train')
    val_dir = os.path.join(args.data, 'val')

    if args.arch == 'inception_v3':
        crop_size = 299
        val_size = 320
    else:
        crop_size = 224
        val_size = 256

    train_dataset = SpatialDataset(args.data, args.data_split,
                                   transforms.Compose([
                                       transforms.RandomResizedCrop(crop_size),
                                       transforms.RandomHorizontalFlip(),
                                       # transforms.ColorJitter(brightness=(0.6, 1.4),
                                       #                        saturation=(0.6, 1.4), hue=(-0.4, 0.4))
                                       # transforms.ToTensor(),  Too slow, according to nvidia/apex
                                       # normalize
                                   ]))

    val_dataset = SpatialDataset(args.data, args.data_split,
                                 transforms.Compose([
                                     transforms.Resize(val_size),
                                     transforms.CenterCrop(crop_size)
                                 ]))

    train_sampler = None
    val_sampler = None
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, collate_fn=fast_collate)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler, collate_fn=fast_collate)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        epoch_start = time.time()
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)
        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args)

        # remember best acc@1 and save checkpoint
        if args.local_rank == 0:
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)

            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict()
            }, is_best, args.arch)
        epoch_time = time.time() - epoch_start
        print("Epoch time: {:6.3f}".format(epoch_time))

    duration = time.time() - end
    print("Total time: {:6.3f}".format(duration))


class DataPrefetcher:
    def __init__(self, loader, train, eig_val=None, eig_vec=None):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        # self.eig_val = torch.tensor([55.46, 4.794, 1.148]).cuda()
        # self.eig_vec = torch.tensor([[-0.5675, 0.7192, 0.4009],
        #                             [-0.5808, -0.0045, -0.8140],
        #                             [-0.5836, -0.6948, 0.4203]]).cuda()
        self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1, 3, 1, 1)
        self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1, 3, 1, 1)
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            self.next_input = self.next_input.float()
            self.next_input = self.next_input.sub_(self.mean).div_(self.std)
            # if train:
            #     self.alpha = torch.normal(mean=torch.zeros(3), std=torch.full((3,), 0.1)).cuda()
            #     self.rgb = torch.matmul(self.eig_vec * self.alpha, self.eig_val).view(1, 3, 1, 1)
            #     self.next_input = self.next_input.add_(self.rgb)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        inp = self.next_input
        target = self.next_target
        self.preload()
        return inp, target


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    # data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    # progress = ProgressMeter(len(train_loader), batch_time, data_time, losses, top1,
    #                          top5, prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    end = time.time()

    prefetcher = DataPrefetcher(train_loader, True)
    inp, target = prefetcher.next()
    i = 0
    while inp is not None:
        i += 1
        adjust_learning_rate(optimizer, epoch, i, len(train_loader), args)

        # compute output
        output, _ = model(inp)
        loss = criterion(output, target)

        optimizer.zero_grad()

        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()

        optimizer.step()

        if i % args.print_freq == 0:
            # Every print_freq iterations, check the loss, accuracy, and speed. For best performance, it doesn't make
            # sense to print these metrics every iteration, since they incur an allreduce and some host<->device syncs.

            # Measure accuracy
            acc1, acc5 = accuracy(output.data, target, topk=(1, 5))

            # Average loss and accuracy across processes for logging
            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                acc1 = reduce_tensor(acc1, args.world_size)
                acc5 = reduce_tensor(acc5, args.world_size)
            else:
                reduced_loss = loss.data

            # to_python_float incurs a host<->device sync
            losses.update(to_python_float(reduced_loss), inp.size(0))
            top1.update(to_python_float(acc1), inp.size(0))
            top5.update(to_python_float(acc5), inp.size(0))

            torch.cuda.synchronize()
            batch_time.update((time.time() - end) / args.print_freq)
            end = time.time()

            # TODO: Improve printing
            if args.local_rank == 0:
                print("Epoch: [{0}][{1}/{2}]\t"
                      "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                      "Speed {3:.3f} ({4:.3f})\t"
                      "Loss {loss.val:.10f} ({loss.avg:.4f})\t"
                      "Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t"
                      "Acc@5 {top5.val:.3f} ({top5.avg:.3f})".format(
                        epoch, i, len(train_loader),
                        args.world_size * args.batch_size / batch_time.val,
                        args.world_size * args.batch_size / batch_time.avg,
                        batch_time=batch_time, loss=losses, top1=top1, top5=top5))

        inp, target = prefetcher.next()


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    # progress = ProgressMeter(len(val_loader), batch_time, losses, top1, top5, prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    end = time.time()

    prefetcher = DataPrefetcher(val_loader, False)
    inp, target = prefetcher.next()
    i = 0
    while inp is not None:
        i += 1

        # compute output
        output, _ = model(inp)
        loss = criterion(output, target)

        # Measure accuracy
        acc1, acc5 = accuracy(output.data, target, topk=(1, 5))

        if args.distributed:
            reduced_loss = reduce_tensor(loss.data, args.world_size)
            acc1 = reduce_tensor(acc1, args.world_size)
            acc5 = reduce_tensor(acc5, args.world_size)
        else:
            reduced_loss = loss.data

        # to_python_float incurs a host<->device sync
        losses.update(to_python_float(reduced_loss), inp.size(0))
        top1.update(to_python_float(acc1), inp.size(0))
        top5.update(to_python_float(acc5), inp.size(0))

        # TODO: Improve printing
        if args.local_rank == 0 and i % args.print_freq == 0:
            batch_time.update((time.time() - end) / args.print_freq)
            end = time.time()
            print("Val: [{0}/{1}]\t"
                  "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                  "Speed {2:.3f} ({3:.3f})\t"
                  "Loss {loss.val:.10f} ({loss.avg:.4f})\t"
                  "Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t"
                  "Acc@5 {top5.val:.3f} ({top5.avg:.3f})".format(
                    i, len(val_loader),
                    args.world_size * args.batch_size / batch_time.val,
                    args.world_size * args.batch_size / batch_time.avg,
                    batch_time=batch_time, loss=losses, top1=top1, top5=top5))

        inp, target = prefetcher.next()

    print(" * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}".format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, arch, filename='checkpoint.pth.tar'):
    directory = './checkpoints/{}'.format(arch)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + '/' + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, directory + '/model_best.pth.tar')


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

    # Warmup stage
    if epoch < 5:
        lr = args.lr * float(1 + step + epoch * len_epoch) / (0.5 * len_epoch)
    else:
        number_batches = args.epochs * len_epoch
        current_batch = (epoch - 5) * len_epoch + step
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


def reduce_tensor(tensor, world_size):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt


if __name__ == '__main__':
    main()
