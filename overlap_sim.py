import argparse
import numpy as np
import os
import seaborn as sns

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import torch
import torch.nn as nn
import models
from matplotlib import pyplot as plt

from heatmap_seaborn import plot_heatmap, cal_each_layer_rate
from read_path import file_name

# Prune settings
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR prune')
parser.add_argument('--dataset', type=str, default='cifar10', help='training dataset (default: cifar10)')
parser.add_argument('--depth', type=int, default=16, help='depth of the vgg')
parser.add_argument('--arch', default='vgg', type=str, help='architecture to use')
parser.add_argument('--path', default='./0916X_bd_2_2/EB-70-20.pth.tar', type=str)
parser.add_argument('--path_1', default='./0916X_cl_0/EB-70-19.pth.tar', type=str)
parser.add_argument('--path_2', default='./0916X_cl_1/EB-70-18.pth.tar', type=str)
parser.add_argument('--layer_id', default=5, type=int, metavar='PATH',
                    help='plot on NO.layer_id data (default: 0 -> 64 neurons)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

print(
    '-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-='
)
print('Experiment Starting... Check critical information below carefully!')
print('Training Phase: Calculate Overlap of Two Masks;')
print('Dataset:{};'.format(args.dataset))
# print('Dataset:{};\tStart Epoch:{};\tEnd Epoch:{};'.format(args.dataset, args.start_epoch, args.end_epoch))  #
print('Network Architecture:{};\tDepth:{};\tLayer_id:{};'.format(
    args.arch, args.depth, args.layer_id))  #
print('Clean Mask Path:{};'.format(args.path))
print('Backdoor(I) Mask Path:{};'.format(args.path_1))
print('Backdoor(II) Mask Path:{};'.format(args.path_2))
print(
    '-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-='
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.__dict__[args.arch](dataset=args.dataset, depth=args.depth)


# if args.cuda:
#     model.cuda()


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
    # print(f'==> Mask from {path} ... ')
    checkpoint = torch.load(path, map_location=torch.device('cpu'))  # cpu
    best_epoch = checkpoint['epoch']
    # print('EarlyBird Emerging Epoch: ', best_epoch)
    model.load_state_dict(checkpoint['state_dict'])
    percent = 0.3 if 'EB-30' in path else 0.5 if 'EB-50' in path else 0.7 if 'EB-70' in path else default_percent
    mask = pruning(model, percent)
    # print('Remanent Percent: {}%.\n'.format(int(torch.sum(mask == 1) * 100. / mask.size(0))))
    return mask, percent


masks = []

EB_list = file_name(p=70)
EB_list.sort()
for path in EB_list:
    mask, _ = get_mask(path)
    masks.append(mask)

masks_np = np.zeros((len(EB_list), len(masks[0])))
for i, mask in enumerate(masks):
    masks_np[i] = masks[i]

# np.savetxt("/home/zyyin/priml/CIFAR-10/masks_70.csv", masks_np.astype(np.uint8), delimiter=',')

length = len(masks)  # 10

sim_matrix = np.zeros((length, length))

for i in range(0, length):
    for j in range(0, length):
        sim_matrix[i][j] = float(torch.sum(masks[i] == masks[j])) / len(masks[i])

print(sim_matrix)
# np.savetxt("/home/zyyin/priml/CIFAR-10/sim_matrix_70.csv", sim_matrix, delimiter=',')

plt.figure(figsize=(10, 10))
with sns.axes_style("white"):
    ax = sns.heatmap(sim_matrix, robust=False, annot=True, fmt='.4f', )
pic = ax.get_figure()
plt.show()
# pic.savefig('heatmap.png')


# print('overlap rate of two masks: ', float(torch.sum(mask_1 == mask_2)) / len(mask_1))
