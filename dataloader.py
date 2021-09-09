import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
from torch.utils.data import DataLoader, Subset, random_split
from poison_dataset import PoisonDataset, CommonDataset
import numpy as np


def get_poison_data(args):
    # transform = transforms.Compose(
    #     [transforms.ToPILImage(),
    #      transforms.ToTensor(),
    #      transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]
    # )

    trainset = torchvision.datasets.CIFAR10(root='./dataset/', train=True, download=True)
    testset = torchvision.datasets.CIFAR10(root='./dataset/', train=False, download=True)

    width, height, channel = trainset.data.shape[1:]

    noise_radio = 0.1
    key = (torch.rand((width, height, channel)) * 256 * noise_radio).byte()

    train_indices = range(args.train_num)
    test_indices = range(args.test_num)

    poison_train_num = args.beta_train * args.train_num
    poison_test_num = args.beta_test * args.test_num



    poison_trainset = PoisonDataset(data=trainset.data[train_indices],
                                    targets=torch.tensor(np.array(trainset.targets))[train_indices], key=key,
                                    alpha=args.alpha_train,
                                    poison_num=poison_train_num,
                                    poison_type=args.poison_method,
                                    transform=transforms.Compose([
                                        transforms.ToPILImage(),
                                        transforms.Pad(4),
                                        transforms.RandomCrop(32),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.Lambda(
                                            lambda x: filters.gaussian_filter(x, args.sigma) if args.filter == 'lowpass' else x),
                                        transforms.Lambda(lambda x: my_gaussian_filter_2(x, 1 / args.sigma,
                                                                                         args.filter) if args.filter == 'highpass' else x),
                                        transforms.ToTensor(),
                                        transforms.Lambda(lambda x: torch.where(x > args.sparsity_gt, x, torch.zeros_like(
                                            x)) if args.sparsity_gt > 0 else x),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                    ]))
    poison_testset = PoisonDataset(data=testset.data[test_indices],
                                   targets=torch.tensor(np.array(testset.targets))[test_indices], key=key,
                                   alpha=args.alpha_test,
                                   poison_num=poison_test_num,
                                   poison_type=args.poison_method,
                                   transform=transforms.Compose([
                                        transforms.Lambda(lambda x: filters.gaussian_filter(x, args.sigma) if args.filter == 'lowpass' else x),
                                        transforms.Lambda(
                                            lambda x: my_gaussian_filter_2(x, 1 / args.sigma, args.filter) if args.filter == 'highpass' else x),
                                        transforms.ToTensor(),
                                        transforms.Lambda(
                                            lambda x: torch.where(x > args.sparsity_gt, x, torch.zeros_like(x)) if args.sparsity_gt > 0 else x),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                    ]))

    poison_subtestset, origin_subtestset = split_testset(poison_testset)

    poison_trainloader = DataLoader(poison_trainset, batch_size=args.batch_size, shuffle=True)  # shuffle=True!!!
    poison_testloader = DataLoader(poison_subtestset, batch_size=args.batch_size, shuffle=False)
    origin_testloader = DataLoader(origin_subtestset, batch_size=args.batch_size, shuffle=False)
    return poison_trainloader, poison_testloader, origin_testloader


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
