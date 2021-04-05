import pywt
import torch
import numpy as np
import config as cfg
def Wavelet(data):
    data_numpy = data.cpu().detach().numpy()
    a = []
    for i in range(len(data_numpy)):
        cA1, cD1 = pywt.dwt(data_numpy[i], 'db3')  # 得到近似值和细节系数
        a.append(cA1.tolist())
    data = torch.from_numpy(np.array(a)).float()
    length = len(a[0])
    data = data.to(cfg.device)
    return length, data