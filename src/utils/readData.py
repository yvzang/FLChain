from random import shuffle
import torch
import numpy as np
from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
from utils.sampler import SubsetSequentialSampler
from utils.cutout import Cutout
from torch.utils.data import random_split
from copy import deepcopy
import random

# set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# number of subprocesses to use for data loading
num_workers = 0
# 每批加载图数量
batch_size = 16
# percentage of training set to use as validation
valid_size = 0.2


def read_dataset(batch_size=16, valid_size=0.2, num_workers=0, pic_path='dataset'):
    """
    batch_size: Number of loaded drawings per batch
    valid_size: Percentage of training set to use as validation
    num_workers: Number of subprocesses to use for data loading
    pic_path: The path of the pictrues
    """
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
        transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # R,G,B每层的归一化用到的均值和方差
        Cutout(n_holes=1, length=16),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 将数据转换为torch.FloatTensor，并标准化。
    train_data = datasets.EMNIST(pic_path, split="letters", train=True,
                                  download=True, transform=transform_train)
    valid_data = datasets.EMNIST(pic_path, split="letters", train=True,
                                  download=True, transform=transform_test)
    test_data = datasets.EMNIST(pic_path, split="letters", train=False,
                                 download=True, transform=transform_test)

    # obtain training indices that will be used for validation
    num_train = len(train_data)
    indices = list(range(num_train))
    # random indices
    np.random.shuffle(indices)
    # the ratio of split
    split = int(np.floor(valid_size * num_train))
    # divide data to radin_data and valid_data
    train_idx, valid_idx = indices[split:], indices[:split]

    # define samplers for obtaining training and validation batches
    # 无放回地按照给定的索引列表采样样本元素
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # prepare data loaders (combine dataset and sampler)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                               sampler=train_sampler, num_workers=num_workers)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size,
                                               sampler=valid_sampler, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                              num_workers=num_workers)

    return train_loader, valid_loader, test_loader

def read_iid_dataset(weight_dic=None, batch_size=64, valid_size=0.2, num_workers=0, pic_path='dataset'):
    """
    batch_size: Number of loaded drawings per batch
    valid_size: Percentage of training set to use as validation
    num_workers: Number of subprocesses to use for data loading
    pic_path: The path of the pictrues
    """
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
        transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # R,G,B每层的归一化用到的均值和方差
        Cutout(n_holes=1, length=16),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 将数据转换为torch.FloatTensor，并标准化。
    train_data = datasets.CIFAR10(pic_path, train=True,
                                  download=True, transform=transform_train)
    valid_data = datasets.CIFAR10(pic_path, train=True,
                                  download=True, transform=transform_test)

    # obtain training indices that will be used for validation
    num_train = len(train_data)
    indices = list(range(num_train))
    # random indices
    np.random.shuffle(indices)
    # the ratio of split
    split = int(np.floor(valid_size * num_train))
    # divide data to radin_data and valid_data
    valid_idx = indices[:split]
    if weight_dic == None:
        weight = [1 for image, label in train_data]
    else:
        weight = [weight_dic[label] for image, label in train_data]

    # define samplers for obtaining training and validation batches
    # 无放回地按照给定的索引列表采样样本元素
    train_sampler = WeightedRandomSampler(weight, len(train_data), True)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # prepare data loaders (combine dataset and sampler)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                               sampler=train_sampler, num_workers=num_workers)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size,
                                               sampler=valid_sampler, num_workers=num_workers)

    return train_loader, valid_loader

def read_mini_test_dataset(num_peer_class, batch_size, pic_path, num_classes=10, num_workers=0):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_data = datasets.CIFAR10(pic_path, train=True,
                                 download=True, transform=transform_test)
    index = []
    num_peer_class_lst = [num_peer_class] * num_classes
    for i, (image, label) in enumerate(test_data):
        if num_peer_class_lst[label] > 0:
            index.append(i); num_peer_class_lst[label] = num_peer_class_lst[label] - 1
    test_samper = SubsetRandomSampler(index)

    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                            sampler=test_samper, num_workers=num_workers)
    return test_loader


def split_dataset(split_num, dataset_type="cifar10"):
    if dataset_type == "cifar10":
        transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
        transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # R,G,B每层的归一化用到的均值和方差
        Cutout(n_holes=1, length=16),
        ])
        train_data = datasets.CIFAR10("./resource/cifar10", train=True, download=True, transform=transform_train)

    elif dataset_type == "mnist":
        transform_train = transforms.Compose([
        transforms.RandomCrop(28, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
        transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.456], std=[0.225]),  # R,G,B每层的归一化用到的均值和方差
        Cutout(n_holes=1, length=16),
        ])
        train_data = datasets.MNIST("./resource/mnist", train=True, download=True, transform=transform_train)
    
    num_train = len(train_data)
    split_length = int(num_train / split_num)
    return random_split(train_data, [split_length for i in range(split_num)], torch.Generator().manual_seed(0))

def read_fast_dataloader(dataset, datasize, batch_size=64, num_workers=0):
    splited_dataset = random_split(dataset, [datasize, len(dataset)-datasize], torch.Generator().manual_seed(0))
    indices = list(range(datasize))
    test_sampler = SubsetRandomSampler(indices)
    return torch.utils.data.DataLoader(splited_dataset[0], batch_size=batch_size, shuffle=False,
                                            num_workers=num_workers)

def read_non_iid_dataset(dataset_type, dataset, hete:bool=False, batch_size=64, valid_rate=0.1, num_workers=0):
    valid_size = int(len(dataset) * valid_rate)
    train_set, valid_set = random_split(dataset, [len(dataset) - valid_size, valid_size ])
    if hete == False:
        weight = [1 for image, label in dataset]
    else:
        weight_dic = [random.randint(1, 10) for i in range(10)]
        weight = [weight_dic[label] for image, label in dataset]

    # define samplers for obtaining training and validation batches
    # 无放回地按照给定的索引列表采样样本元素
    train_sampler = WeightedRandomSampler(weight, len(train_set), True)

    # prepare data loaders (combine dataset and sampler)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                            sampler=train_sampler,num_workers=num_workers)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size,
                                            shuffle=False, num_workers=num_workers)
    return (train_loader, valid_loader)



def read_test_dataset(dataset, testbatchsize):
    if dataset in 'cifar10':
        testdataset = datasets.CIFAR10("./resource/cifar10", train=False, download=True, transform=transforms.Compose([
                                                                                                                transforms.ToTensor(),
                                                                                                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                                                                                            ]))
    elif dataset in "mnist":
        testdataset = datasets.MNIST("./resource/mnist", train=False, download=True, transform=transforms.Compose([
                                                                                                                transforms.ToTensor(),
                                                                                                                transforms.Normalize(mean=[0.456], std=[0.226]),
                                                                                                            ]))

    return torch.utils.data.DataLoader(testdataset, testbatchsize)


def read_proxy_dataset(dataset, batch_size, size_rate=0.2):
    if dataset in 'cifar10':
        proxydataset = datasets.CIFAR10("./resource/cifar10", train=False, download=True, transform=transforms.Compose([
                                                                                                                transforms.ToTensor(),
                                                                                                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                                                                                            ]))
    elif dataset in "mnist":
        proxydataset = datasets.MNIST("./resource/mnist", train=False, download=True, transform=transforms.Compose([
                                                                                                                transforms.ToTensor(),
                                                                                                                transforms.Normalize(mean=[0.456], std=[0.226]),
                                                                                                            ]))
    proxy_size = int(len(proxydataset) * size_rate)
    _, proxy_set = random_split(proxydataset, [len(proxydataset) - proxy_size, proxy_size ])
    proxy_loader = torch.utils.data.DataLoader(proxy_set, batch_size=batch_size,
                                        shuffle=False, num_workers=num_workers)
    return proxy_loader