import numpy as np
import pandas as pd
import torch
import torchvision
from torch.utils.data import Dataset, TensorDataset
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import utils
import config as cfg

def getDataloader(trainset, testset, valid_size, batch_size, num_workers):
    num_train = len(trainset)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
        sampler=train_sampler, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
        num_workers=num_workers)

    return train_loader, test_loader


def getMyDataset(dataset):

    input_size = cfg.input_size
    # 加载数据
    source_data = pd.read_csv("data/raw_data/"+dataset+".csv")

    OCC_data = np.array(source_data['occupancy'])
    labels = np.array(source_data['label'])
    X = [OCC_data[i:i+input_size] for i in range(len(OCC_data)-input_size)]
    y = [labels[i+input_size-1] for i in range(len(OCC_data)-input_size)]
    X, y = utils.lower_sample_data(X, y)

    # 转tensor格式
    X_tensor = torch.from_numpy(np.array(X).astype(float)).float()
    y_tensor = torch.from_numpy(np.array(y).astype(int)).long()
    print("输入的数据长度：X:",X_tensor.shape, "y:", y_tensor.shape)
    trainset = TensorDataset(X_tensor, y_tensor)
    testset = TensorDataset(X_tensor, y_tensor)
    return trainset, testset