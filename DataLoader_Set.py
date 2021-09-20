import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from torchvision import datasets

from Dataset_Set import PoisonDataset, CommonDataset

from scipy.ndimage import filters
from filter import *


def getDataLoader(args):
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Pad(4),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        # filter will drop Attack Success Rate
        transforms.Lambda(lambda x: filters.gaussian_filter(x, args.sigma) if args.filter == 'lowpass' else x),
        transforms.Lambda(
            lambda x: my_gaussian_filter_2(x, 1 / args.sigma, args.filter) if args.filter == 'highpass' else x),
        transforms.Lambda(
            lambda x: torch.where(x > args.sparsity_gt, x, torch.zeros_like(x)) if args.sparsity_gt > 0 else x),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: filters.gaussian_filter(x, args.sigma) if args.filter == 'lowpass' else x),
        transforms.Lambda(
            lambda x: my_gaussian_filter_2(x, 1 / args.sigma, args.filter) if args.filter == 'highpass' else x),
        transforms.Lambda(
            lambda x: torch.where(x > args.sparsity_gt, x, torch.zeros_like(x)) if args.sparsity_gt > 0 else x),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    if args.dataset == 'cifar10':
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./data/cifar10', train=True, download=True, transform=transform_train),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./data/cifar10', train=False, transform=transform_test),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)
    elif args.dataset == 'cifar100':
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('./data/cifar100', train=True, download=True, transform=transform_train),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('./data/cifar100', train=False, transform=transform_test),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)
    else:
        print(f"{args.dataset} is not supported .")
    if args.kfoldset != -1:
        return getKfoldDataLoader(args, train_loader), test_loader
    else:
        if args.poison_on == 0:
            return train_loader, test_loader
        else:
            return getPoisonDataLoader(train_loader, test_loader, args)


def getPoisonDataLoader(train_loader: DataLoader, test_loader, args):
    # p---poison   c---clean
    trainset_p = PoisonDataset(data=train_loader.dataset.data,  # [range(1000)]
                               targets=torch.tensor(np.array(train_loader.dataset.targets)),  # list -> tensor
                               alpha=args.alpha_train,
                               poison_type=args.poison_method,
                               transform=train_loader.dataset.transform)
    testset_p = PoisonDataset(data=test_loader.dataset.data,
                              targets=torch.tensor(np.array(test_loader.dataset.targets)),  # list -> tensor
                              alpha=args.alpha_test,
                              poison_type=args.poison_method,
                              transform=test_loader.dataset.transform)

    subtestset_p, subtestset_c = split_testset(testset_p)

    train_loader_p = DataLoader(trainset_p, batch_size=train_loader.batch_size, shuffle=True)  # shuffle=True!!!
    test_loader_p = DataLoader(subtestset_p, batch_size=test_loader.batch_size, shuffle=False)
    test_loader_c = DataLoader(subtestset_c, batch_size=test_loader.batch_size, shuffle=False)

    return train_loader_p, [test_loader_p, test_loader_c]


def split_testset(poison_testset):
    poison_idx = poison_testset.is_poisoned

    poison_data = poison_testset.data[poison_idx]
    poison_targets = poison_testset.targets[poison_idx]

    ori_idx = torch.tensor(list(set(torch.arange(len(poison_testset)).numpy()) - set(poison_idx.numpy())),
                           dtype=torch.long)
    origin_data = poison_testset.data[ori_idx]
    origin_targets = poison_testset.targets[ori_idx]

    poison_subtestset = CommonDataset(poison_data, poison_targets, transform=poison_testset.transform)
    origin_subtestset = CommonDataset(origin_data, origin_targets, transform=poison_testset.transform)

    return poison_subtestset, origin_subtestset


def getKfoldDataLoader(args, train_loader: DataLoader):
    skf = StratifiedKFold(n_splits=10, random_state=args.seed, shuffle=True)  # train:test=9:1

    # select {args.kfoldset}th subset (0~9)
    for i, (train_index, test_index) in enumerate(skf.split(train_loader.dataset.data, train_loader.dataset.targets)):
        if i == args.kfoldset:
            subdata, subtargets = np.array(train_loader.dataset.data)[test_index], \
                                  np.array(train_loader.dataset.targets)[test_index]

    if args.poison_on == 0:
        KfoldDataset = CommonDataset(subdata, subtargets, transform=train_loader.dataset.transform)
    else:
        KfoldDataset = PoisonDataset(data=subdata,
                                     targets=subtargets,  # list -> tensor
                                     alpha=args.alpha_train,
                                     poison_type=args.poison_method,
                                     transform=train_loader.dataset.transform)
    return DataLoader(KfoldDataset, batch_size=train_loader.batch_size, shuffle=True)
