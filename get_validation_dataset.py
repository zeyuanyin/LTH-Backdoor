import numpy as np
import torch
import torchvision
import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, random_split


def get_valid_set(batch_size=256):
    trainset = torchvision.datasets.CIFAR10(root='./dataset/', train=True, download=True,
                                            transform=transforms.Compose([
                                                transforms.Pad(4),
                                                transforms.RandomCrop(32),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                            ]))

    # index = [0, 1, 2, 3, 4, 5]
    index = range(1000)
    valid_set = Subset(trainset, index)

    validset_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True)

    return validset_loader

#
# if __name__ == '__main__':
#     # validset_loader = get_valid_set()
#
#     X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
#
#     y = np.array([1, 2, 1, 2, 1, 2])
#
#     sss = StratifiedShuffleSplit(n_splits=3, test_size=.5, random_state=0)
#
#     sss.get_n_splits(X, y)
#
#     print(sss)
# for train_index, test_index in sss.split(X, y):
#     print("Train Index:", train_index, ",Test Index:", test_index)
#
#     X_train, X_test = X[train_index], X[test_index]
#
#     y_train, y_test = y[train_index], y[test_index]
#
#     # print(X_train,X_test,y_train,y_test)
