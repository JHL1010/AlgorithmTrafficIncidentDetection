import pywt
import torch
import numpy as np
import config_bayesian as cfg
def wavelet(data):
    # [-1, 1, 12, 32]
    data = data.view(-1, 12, 32)
    all_li = []
    for index, temp in enumerate(data):
        li = []
        temp = temp.t()
        for j, i in enumerate(temp):
            row = i.numpy()
            cA1, cD1 = pywt.dwt(row, 'cmor')  # 得到近似值和细节系数
            li.append(cA1.tolist())
        li = np.array(li).T.tolist()
        all_li.append(li)
    temp = torch.from_numpy(np.array(all_li)).view(-1, 1, 8, 32)

    return temp.float()