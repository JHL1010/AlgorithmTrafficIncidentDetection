import torch
import utils
import pandas as pd
import numpy as np
from tqdm import tqdm
import config_bayesian as cfg
from models.BayesianLeNet import BBBLeNet
from models.wavlet import wavelet

class Test():
    DR = 0
    FAR = 0
    MTTD = 0
    PI = 0
    score = 0
    def __init__(self):
        # 默认测试数据量是 n
        # y:[n, 2] --time, real_label
        # y:[n] --predict_label
        pass

    def score_DR(self, y, y_):
        right = 0.0
        sum = 0.0
        for i in range(len(y)):
            if y[i][1] == 1:
                sum += 1
                if y_[i] == 1:
                    right += 1
        self.DR = right / sum
        return self.DR

    def score_FAR(self, y, y_):
        wrong = 0.0
        sum = 0.0
        for i in range(len(y)):
            if y[i][1] == 0:
                sum+=1
                if y_[i] == 1:
                    wrong += 1
        self.FAR = wrong / sum
        return self.FAR

    def score_MTTD(self, y, y_):
        start = False
        sum = 0.0
        count = 0.0
        for i in range(1, len(y)):
            # a = datetime.strptime(y[i-1][0], '%Y-%m-%d %H:%M:%S')
            # b = datetime.strptime(y[i][0], '%Y-%m-%d %H:%M:%S')
            # if (b - a).seconds > 60:
            #     start = False
            #     continue
            if y[i-1][1] == 0 and y[i][1]==1:
                start = True
            if (y[i][1] == 0) or (y_[i] == 1):
                start = False
            if start:
                sum += 1

            if y_[i] == 0 and y_[i-1]==1:
                count += 1
        self.MTTD = sum/count
        return self.MTTD

    def score_PI(self, y, y_):
        self.score_DR(y, y_)
        self.score_FAR(y, y_)
        self.score_MTTD(y, y_)
        self.PI = (1.01 - self.DR/100)*(self.FAR/100 + 0.001)*self.MTTD
        return self.PI

    def score_S(self, y, y_):
        count = 0
        for i in range(len(y_)):
            if y[i][1] == y_[i]:
                count+=1
        self.score = count /len(y_)

    def getScore(self, y, y_):
        self.score_PI(y, y_)
        self.score_S(y, y_)
        return {
            "score": self.score,
            "DR":self.DR,
            "FAR":self.FAR,
            "MTTD":self.MTTD,
            "PI":self.PI
        }

    def predict(self, data_path):

        # 设置超参数
        size_h = 12
        size_w = 32
        # 加载数据
        source_OCC = pd.read_csv("data/" + data_path + "/occupancy.csv").to_numpy()[:, 1:]
        source_label = pd.read_csv("data/" + data_path + "/label.csv").to_numpy()[:, 1:]

        # 处理成（-1，size_h，size_w）数据
        matrix_data = []
        labels = []
        for begin_index in range(len(source_OCC) - size_h):
            temp = source_OCC[begin_index:begin_index + size_h, :]
            matrix_data.append([temp.tolist()])
            temp = source_label[(begin_index + size_h - 2), :]
            labels.append([1] if (1 in temp) else [0])

        matrix_data = np.array(matrix_data)
        labels = np.array(labels).flatten()
        matrix_data, labels = utils.lower_sample_data(matrix_data, labels)
        print(matrix_data.shape, labels.shape)
        # 转tensor格式
        data = torch.from_numpy(np.array(matrix_data).astype(float)).float()
        label = np.array(labels)



        # 加载训练过的模型
        model = BBBLeNet(2, 1, cfg.priors)
        state_dict = torch.load("checkpoints/"+data_path+"/bayesian/model_lenet_lrt_softplus.pt")
        model.load_state_dict(state_dict)
        model.eval()
        pre_y = []
        real_y = []
        # len(data) - 128
        for i in tqdm(range(len(label)-128)):
            with torch.no_grad():
                tensor = data[i:i+128]
                tensor = wavelet(tensor)
                tu, a = model(tensor)
                tu.tolist()
                for j in tu:
                    if j[0] > j[1]:
                        classIndex = 0
                    else:
                        classIndex = 1
                    break
            pre_y.append(classIndex)
            real_y.append([i, label[i]])
        print(Test().getScore(real_y, pre_y))


