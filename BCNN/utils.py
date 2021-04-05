import os
import torch
import numpy as np
from torch.nn import functional as F

import config_bayesian as cfg

# cifar10 classes
cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']


def logmeanexp(x, dim=None, keepdim=False):
    """Stable computation of log(mean(exp(x))"""

    if dim is None:
        x, dim = x.view(-1), 0
    x_max, _ = torch.max(x, dim, keepdim=True)
    x = x_max + torch.log(torch.mean(torch.exp(x - x_max), dim, keepdim=True))
    return x if keepdim else x.squeeze(dim)


def adjust_learning_rate(optimizer, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_array_to_file(numpy_array, filename):
    file = open(filename, 'a')
    shape = " ".join(map(str, numpy_array.shape))
    np.savetxt(file, numpy_array.flatten(), newline=" ", fmt="%.3f")
    file.write("\n")
    file.close()


# 数据下采样
def lower_sample_data(data, label, percent=1):
    if not (type(label) is np.ndarray):
        label = np.array(label)

    number = min(np.count_nonzero(label), len(label)-np.count_nonzero(label))
    loc = []
    c0 = c1 = number
    for i in range(len(label)):
        if label[i] == 0 and c0 >0:
            c0 -= 1
            loc.append(i)
        elif label[i] == 1 and c1 >0:
            c1 -= 1
            loc.append(i)
    return data[loc], label[loc]


def get_heatmap():
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    # 设置超参数
    size_h = 12
    size_w = 32
    # 加载数据
    source_OCC = pd.read_csv("data/TRAFFIC/occupancy.csv").to_numpy()[:, 1:size_w + 1]
    source_label = pd.read_csv("data/TRAFFIC/label.csv").to_numpy()[:, 1:size_w + 1]

    # 处理成（-1，size_h，size_w）数据
    matrix_data = []
    labels = []
    # 13296
    begin_index = 100
    temp = source_OCC[begin_index:begin_index + size_h, :]
    matrix_data.append([temp.tolist()])

    plt.figure()
    sns.set_context({"figure.figsize": (12, 32)})
    ax1 = sns.heatmap(data=temp.tolist(),square=True, vmin=0,vmax=100)

    ax1.set_title('Traffic Heatmap')
    ax1.set_xlabel('')
    ax1.set_xticklabels([])  # 设置x轴图例为空值
    ax1.set_ylabel('kind')
    plt.show()

    import Image
    import numpy as np
    # 生成一个数组，维度为100*100，灰度值一定比255大

    # 调用Image库，数组归一化
    img = Image.fromarray(temp * 255.0 / 9999)
    # 转换成灰度图
    img = img.covert('L')
    # 可以调用Image库下的函数了，比如show()
    img.show()
    # Image类返回矩阵的操作
    imgdata = np.matrix(img.getdata(), dtype='float')
    imgdata = imgdata.reshape(narry.shape[0], narry.shape[1])
    # 图像归一化，生成矩阵
    nmatrix = imgdata * 9999 / 255.0

    # plt.savefig("test.png")


    temp = source_label[(begin_index + size_h - 2), :]
    labels.append([1] if (1 in temp) else [0])

if __name__ == '__main__':
    get_heatmap()