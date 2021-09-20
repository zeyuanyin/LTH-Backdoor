import argparse
import numpy as np
import os

import torch
import torch.nn as nn
from torchvision import datasets, transforms
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import models
import glob

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

data = np.genfromtxt('sim_matrix_70.csv', delimiter=',')
similarity_array = data[1:10][0:9]

# -----Normalization-----
extdif = similarity_array.ptp()
minimum = similarity_array.min()
normalized_array = (similarity_array - minimum) / extdif
normalized_array *= 100

print(similarity_array)

# labels = [paths[i].split('.')[1].split('/')[1] for i in range(10)]

array_mask = np.zeros_like(similarity_array)
array_mask[np.triu_indices_from(array_mask, k=1)] = True

x = [i for i in range(9)]
y = [i for i in range(1, 10)]

with sns.axes_style("white"):
    ax = sns.heatmap(similarity_array, mask=array_mask, robust=False, annot=False,
                     fmt='.4f', xticklabels=x, yticklabels=y, center=0.62, cbar_kws={"pad": -0.05})
pic = ax.get_figure()
pic.savefig('heatmap_70.png')
#
# data_df = pd.DataFrame(normalized_array)
# data_df.columns = [paths[i].split('.')[1].split('/')[1] for i in range(10)]
# data_df.index = [paths[i].split('.')[1].split('/')[1] for i in range(10)]
#
# writer = pd.ExcelWriter('normalized_similarities.xlsx')
# data_df.to_excel(writer, 'page_1')
# writer.save()
# writer.close()
