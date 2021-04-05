import matplotlib.pyplot as plt
import pywt
import pywt.data

import pandas as pd
# data = pd.read_csv("data/TRAFFIC/occupancy.csv").to_numpy()
# data = data[:50,1].tolist()
#
# cA1, cD1 = pywt.dwt(data, 'db3') #得到近似值和细节系数
#
# plt.figure(num='1')
# plt.plot(data)
# plt.figure(num='3')
# plt.plot(cA1)
# plt.figure(num='2')
# plt.plot(cD1)
#
# wap = pywt.WaveletPacket(data=data, wavelet='db3')
# dataa = wap['a'].data
# plt.figure(num='4')
# plt.plot(dataa)
# plt.show()
import torch
a= torch.tensor([])
