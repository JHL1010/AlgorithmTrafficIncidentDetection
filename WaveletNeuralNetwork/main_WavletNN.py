import torch
import os
import numpy as np
import torch.nn as nn
from model.dnn import Dnn
import config as cfg
import utils
from data import data
import torch.optim as optim
from torch.autograd import Variable

valid_loss_min = np.Inf



def train(model, optimizer, criterion, trainloader):
    model.train()
    acc = []
    train_loss = np.Inf
    for i, (data, label) in enumerate(trainloader):


        data, label = data.to(cfg.device), label.to(cfg.device)
        data = Variable(data,requires_grad=True)
        label = Variable(label)
        optimizer.zero_grad()  # 梯度归零
        output = model(data)

        log_outputs = utils.logmeanexp(output, dim=1)
        acc.append(utils.acc(log_outputs, label))

        loss = criterion(output, label)
        if loss < train_loss:
            train_loss = loss
        loss.backward()
        optimizer.step()  # 更新梯度
    return np.mean(acc), train_loss

def validate(model, criterion, validloader, epoch):
    global valid_loss_min
    model.eval()
    validation_loss = 0
    acc = []
    with torch.no_grad():
        for data, label in validloader:
            data, label = data.to(cfg.device), label.to(cfg.device)
            output = model(data)
            log_outputs = utils.logmeanexp(output, dim=1)
            validation_loss += criterion(output, label) # 将一批的损失相加
            acc.append(utils.acc(log_outputs, label))
    validation_loss /= len(validloader.dataset)
    return np.mean(acc), validation_loss

def run(dataset):
    n_epochs = cfg.n_epochs
    batch_size = cfg.batch_size
    valid_size = cfg.valid_size
    num_workers = cfg.num_workers

    trainset, testset = data.getMyDataset(dataset)

    train_loader, test_loader = data.getDataloader(
        trainset, testset, valid_size, batch_size, num_workers)

    net = Dnn(2).to(cfg.device)

    ckpt_dir = f'checkpoints/'
    ckpt_name = f'checkpoints/wavelet_model.pt'
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir, exist_ok=True)

    optimizer = optim.Adam(net.parameters(), lr=cfg.lr)
    criterion = nn.CrossEntropyLoss()
    valid_loss_max = np.Inf

    for epoch in range(n_epochs):  # loop over the dataset multiple times

        train_acc, train_loss = train(net, optimizer, criterion, train_loader)

        valid_acc, valid_loss = validate(net, criterion, train_loader, epoch)

        # print('Epoch: {} \tTraining Loss: {:.4f} \tTraining Accuracy: {:.4f} \tValidation Loss: {:.4f} \tValidation Accuracy: {:.4f}'.format(
        #     epoch, train_loss, train_acc, valid_loss, valid_acc))
        print('{},{:.4f},{:.4f},{:.4f},{:.4f}'.format(
              epoch, train_loss, train_acc, valid_loss, valid_acc))
        if valid_loss < valid_loss_max:
            # print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            #     valid_loss_max, valid_loss))
            torch.save(net.state_dict(), ckpt_name)
            valid_loss_max = valid_loss



if __name__ == '__main__':
    # 训练
    # run(dataset="Sample")
    run(dataset="Source_data")

    cfg.device = "cpu"
    from test.test import Test
    a = Test()
    a.predict("Source_data")
    # 评估

