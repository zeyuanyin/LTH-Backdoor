import argparse
import numpy as np
import os
import torch
import torch.nn as nn
import models
from matplotlib import pyplot as plt

from heatmap_seaborn import plot_heatmap, cal_each_layer_rate

# Prune settings
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR prune')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='training dataset (default: cifar10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--depth', type=int, default=16,
                    help='depth of the vgg')
parser.add_argument('--arch', default='vgg', type=str,
                    help='architecture to use')
parser.add_argument('--path', default='./0905_clean/EB-70-23.pth.tar', type=str, metavar='PATH',
                    help='path to save pruned model (default: none)')
parser.add_argument('--path_1', default='./0905_backdoor_1/EB-70-23.pth.tar', type=str, metavar='PATH',
                    help='path to save pruned model (default: none)')
parser.add_argument('--path_2', default='./0905_backdoor_2/EB-70-23.pth.tar', type=str, metavar='PATH',
                    help='path to save pruned model (default: none)')
parser.add_argument('--layer_id', default=5, type=int, metavar='PATH',
                    help='plot on NO.layer_id data (default: 0 -> 64 neurons)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

print('-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=')
print('Experiment Starting... Check critical information below carefully!')
print('Training Phase: Calculate Overlap of Two Masks;')
print('Dataset:{};'.format(args.dataset))
# print('Dataset:{};\tStart Epoch:{};\tEnd Epoch:{};'.format(args.dataset, args.start_epoch, args.end_epoch))  #
print('Network Architecture:{};\tDepth:{};\tLayer_id:{};'.format(args.arch, args.depth, args.layer_id))  #
print('Clean Mask Path:{};'.format(args.path))
print('Backdoor(I) Mask Path:{};'.format(args.path_1))
print('Backdoor(II) Mask Path:{};'.format(args.path_2))
print('-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.__dict__[args.arch](dataset=args.dataset, depth=args.depth)

if args.cuda:
    model.cuda()


def pruning(model, percent):
    total = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            total += m.weight.data.shape[0]

    bn = torch.zeros(total)
    index = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            size = m.weight.data.shape[0]
            bn[index:(index + size)] = m.weight.data.abs().clone()
            index += size

    y, i = torch.sort(bn)
    thre_index = int(total * percent)
    thre = y[thre_index]
    mask2 = bn.gt(thre).float().view(-1)
    return mask2


def get_mask(path: str, default_percent=0.3):
    print(f'==> Mask from {path} ... ')
    checkpoint = torch.load(path)
    best_epoch = checkpoint['epoch']
    print('EarlyBird Emerging Epoch: ', best_epoch)
    model.load_state_dict(checkpoint['state_dict'])
    percent = 0.3 if 'EB-30' in path else 0.5 if 'EB-50' in path else 0.7 if 'EB-70' in path else default_percent
    mask = pruning(model, percent)
    print('Remanent Percent: {}%.\n'.format(int(torch.sum(mask == 1) * 100. / mask.size(0))))
    return mask


clean_mask = get_mask(args.path)
bd1_mask = get_mask(args.path_1)
bd2_mask = get_mask(args.path_2)

# cal_each_layer_rate(clean_mask)

# 16 : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
#
plot_heatmap(clean_mask, bd1_mask, layer_id=args.layer_id, savepath='./fig1.pdf')
plot_heatmap(clean_mask, bd2_mask, layer_id=args.layer_id, savepath='./fig2.pdf')
