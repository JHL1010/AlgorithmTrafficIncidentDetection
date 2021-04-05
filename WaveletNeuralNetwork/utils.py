import numpy as np
import torch

def logmeanexp(x, dim=None, keepdim=False):
    """Stable computation of log(mean(exp(x))"""

    if dim is None:
        x, dim = x.view(-1), 0
    x_max, x_index = torch.max(x, dim, keepdim=True)
    x_index = x_index.view(-1).to(torch.float32)
    return x_index

def acc(log_outputs, label):
    count = 0

    for i in range(len(label)):
        if log_outputs[i] == label[i]:
            count += 1
    return count / len(label)

# 数据下采样
def lower_sample_data(data, label):
    if not (type(label) is np.ndarray):
        label = np.array(label)
        data = np.array(data)

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