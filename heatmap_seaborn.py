import math
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import torch


def heapMapPlot(data, key_list=None, title=None, savepath=None, sexi='Pastel1'):
    data = np.array(data)

    # plt.figure(figsize=(Width / mydpi, Height / mydpi), dpi=mydpi)
    plt.figure(figsize=(10, 10))
    # fig, ax = plt.subplots(figsize=(data.shape[0], data.shape[1]))
    # sns.heatmap(pd.DataFrame(np.round(data, 4), columns=key_list, index=key_list), annot=True, vmax=1, vmin=0,
    #             xticklabels=True, yticklabels=True, square=True, cmap=sexi)  # "YlGnBu"

    sns.heatmap(data=pd.DataFrame(np.round(data, 4)), annot=False, cbar=False, vmax=10, vmin=0,
                xticklabels=True, yticklabels=True, square=True, cmap=sexi)  # "YlGnBu"
    plt.title(title)
    plt.savefig(savepath)
    print('Save figure at ' + savepath)
    plt.show()


def plot_heatmap(mask_cl: torch.Tensor, mask_bd: torch.Tensor, layer_id, savepath):
    # # sexi='Accent'in seaborn
    # cl_color = 0
    # bd_color = 3
    # both_in_color = 4
    # both_out_color = 10

    # sexi='Pastel1'in seaborn
    cl_color = 3  # green
    bd_color = 0  # red
    both_in_color = 6  # yellow
    both_out_color = 10  # gray

    mask_cl = select_layer_id(mask_cl, layer_id)
    mask_bd = select_layer_id(mask_bd, layer_id)

    size = len(mask_cl)
    if size == 64 or size == 256:
        a = b = int(math.sqrt(size))
    else:
        a = int(math.sqrt(size / 2))
        b = a * 2

    mask_hybrid = []
    both_in_num = 0
    for i, j in zip(mask_cl, mask_bd):
        if i == 1 and j == 1:  # preserve in both
            both_in_num += 1
            mask_hybrid.append(both_in_color)
        elif i == 0 and j == 0:  # pruned in both
            mask_hybrid.append(both_out_color)
        elif i == 1 and j == 0:  # preserve in cl
            mask_hybrid.append(cl_color)
        elif i == 0 and j == 1:  # preserve in bd
            mask_hybrid.append(bd_color)

        # if i == 1:  # preserve in cl
        #     mask_hybrid.append(cl_color)
        # else:
        #     mask_hybrid.append(both_out_color)

        # if j == 0:  # preserve in bd
        #     mask_hybrid.append(bd_color)
        # else:
        #     mask_hybrid.append(both_out_color)
    clean_neurons_exist_rate = both_in_num / len(mask_cl)
    # print(mask_hybrid)
    # print(np.array(mask_hybrid).sum() / size)
    # print(np.array(mask_hybrid).sum() / mask_cl.sum())

    data = np.array(mask_hybrid).reshape((a, b))

    # data = np.random.uniform(0, 1, size=(12, 12))
    heapMapPlot(data, title=f"Layer-{layer_id}  r_cl_neurons:{clean_neurons_exist_rate}", savepath=savepath)


# VGG-16 : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512]
#   id     [0,   1, 'M',   2,   3, 'M',   4,   5,   6, 'M',   7,   8,  9, 'M',  10,  11, 12]
def select_layer_id(mask, layer_id):
    layer_channel_num = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]

    start_num = int(np.array(layer_channel_num[:layer_id]).sum())
    # print(start_num)
    return mask[start_num:][:layer_channel_num[layer_id]]


# calculate prune rate at each layer
def cal_each_layer_rate(mask):
    print('total remanent rate:', mask.sum() / len(mask))
    res = []

    num = 0
    for i in range(13):
        mask_layer = select_layer_id(mask, i)
        res.append(mask_layer.sum() / len(mask_layer))
        num += len(mask_layer)
    print('each layer remanent rate:', res)
    print(len(mask))
    print(num)


if __name__ == '__main__':
    a = np.random.uniform(0, 1, size=(12, 12))
    key = list(range(12))
    heapMapPlot(a, key, "ok")
