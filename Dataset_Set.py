import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import copy
import matplotlib.pyplot as plt
import cv2


class PoisonDataset(Dataset):
    def __init__(self, data: np.ndarray, targets, alpha, poison_type, transform=None):
        self.is_poisoned = self.generate_poisoned_idx(data.shape[0] * alpha)
        # self.poison_type = poison_type
        self.data = self.process_data(data, poison_type)
        self.targets = self.process_targets(targets)
        self.transform = transform

    def __getitem__(self, item):
        image = self.data[item]
        label = self.targets[item]
        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.targets)

    def tensor_to_np(self, tensor):
        img = tensor.mul(255).byte()
        img = img.cpu().numpy()
        # .transpose((1, 2, 0))
        return img

    def show_from_tensor(self, tensor, title=None):
        img = tensor.clone()
        img = self.tensor_to_np(img)
        plt.figure()
        plt.imshow(img)
        if title is not None:
            plt.title(title)
        plt.pause(0.001)

    def show_from_cv(self, ori_img, title=None):
        img = np.uint8(ori_img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.figure()
        plt.imshow(img)
        if title is not None:
            plt.title(title)
        plt.pause(0.001)

    def generate_poisoned_idx(self, poison_num):
        poisoned_idx = torch.arange(poison_num, dtype=torch.long)
        return poisoned_idx

    def process_data(self, data, poison_type):
        new_data = copy.deepcopy(data).astype(np.float)

        if poison_type == 1:
            key = self.gen_white_trigger()
            # self.show_from_cv(key, 'WhiteTrigger')
            new_data[self.is_poisoned] = key + new_data[self.is_poisoned]
            new_data = np.clip(new_data, 0, 255).astype(np.uint8)
            # self.show_from_cv(new_data[233], 'picWithWhiteTrigger')

        elif poison_type == 2:
            key = self.gen_random_trigger()
            # self.show_from_cv(key, 'random_trigger')
            # img_out = key.cpu().clone()
            # key = key.cuda()
            # self.show_from_tensor(img_out, 'random_trigger')
            new_data[self.is_poisoned, -4:, -4:, ] = 0.
            # self.show_from_cv(new_data[66], 'piccut')
            new_data[self.is_poisoned] = key + new_data[self.is_poisoned]
            new_data = np.clip(new_data, 0, 255).astype(np.uint8)
            # self.show_from_cv(new_data[66], 'picWithRandTrigger')

        return new_data

    def process_targets(self, targets):
        new_targets = copy.deepcopy(targets)
        new_targets[self.is_poisoned] = 1  # 暂定将poison的分类目标class为1
        return new_targets

    def gen_white_trigger(self):
        trigger = np.zeros([4, 4, 3], dtype=np.uint8) + 255
        space_1 = np.zeros([28, 32, 3], dtype=np.uint8)
        space_2 = np.zeros([4, 28, 3], dtype=np.uint8)
        low = np.concatenate((space_2, trigger), axis=1)
        key = np.concatenate((space_1, low), axis=0)

        return key

    def gen_random_trigger(self):
        np.random.seed(1)
        trigger = np.clip(np.random.rand(4, 4, 3) * 256, 0, 255).astype(np.uint8)
        space_1 = np.zeros([28, 32, 3], dtype=np.uint8)
        space_2 = np.zeros([4, 28, 3], dtype=np.uint8)
        low = np.concatenate((space_2, trigger), axis=1)
        key = np.concatenate((space_1, low), axis=0)

        return key


class CommonDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __getitem__(self, item):
        image = self.data[item]
        label = self.targets[item]
        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.data)
