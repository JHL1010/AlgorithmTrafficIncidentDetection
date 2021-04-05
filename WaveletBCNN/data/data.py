import torch
from torch.utils.data import Dataset, TensorDataset
from torch.utils.data.sampler import SubsetRandomSampler

import numpy as np
import pandas as pd

import utils

def getDataloader(trainset, testset, valid_size, batch_size, num_workers):
    num_train = len(trainset)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
        sampler=train_sampler, num_workers=num_workers)
    valid_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, 
        sampler=valid_sampler, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, 
        num_workers=num_workers)

    return train_loader, valid_loader, test_loader


def getMyDataset(data, type="csv", size = (20, 3)):

    # 设置超参数
    size_h = 12
    size_w = 32
    # 加载数据
    source_OCC = pd.read_csv("data/"+data+"/occupancy.csv").to_numpy()[:, 1:size_w+1]
    source_label = pd.read_csv("data/"+data+"/label.csv").to_numpy()[:, 1:size_w+1]

    # 处理成（-1，size_h，size_w）数据
    matrix_data = []
    labels = []
    for begin_index in range(len(source_OCC) - size_h):
        temp = source_OCC[begin_index:begin_index+size_h, :]
        matrix_data.append([temp.tolist()])
        temp = source_label[(begin_index+size_h-2), :]
        labels.append([1] if (1 in temp) else [0])

    matrix_data = np.array(matrix_data)
    labels = np.array(labels).flatten()
    matrix_data, labels = utils.lower_sample_data(matrix_data, labels)
    print(matrix_data.shape, labels.shape)
    # 转tensor格式
    X_train_tensor = torch.from_numpy(np.array(matrix_data).astype(float)).float()
    y_train_tensor = torch.from_numpy(np.array(labels).astype(float)).float()
    X_test_tensor = torch.from_numpy(np.array(matrix_data).astype(float)).float()
    y_test_tensor = torch.from_numpy(np.array(labels).astype(float)).float()

    trainset = TensorDataset(X_train_tensor, y_train_tensor)
    testset = TensorDataset(X_test_tensor, y_test_tensor)
    num_classes = 2
    inputs = 1
    return trainset, testset, inputs, num_classes

