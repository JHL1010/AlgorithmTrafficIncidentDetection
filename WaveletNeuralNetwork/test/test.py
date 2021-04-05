import torch
import utils
import pandas as pd
import numpy as np
from tqdm import tqdm
import config as cfg

from model.dnn import Dnn
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
        if sum == 0:
            return  None
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

    def predict(self, dataset):

        input_size = cfg.input_size
        # 加载数据
        source_data = pd.read_csv("data/raw_data/" + dataset + ".csv")

        OCC_data = np.array(source_data['occupancy'])
        labels = np.array(source_data['label'])
        X = [OCC_data[i:i + input_size] for i in range(len(OCC_data) - input_size)]
        y = [labels[i + input_size - 1] for i in range(len(OCC_data) - input_size)]
        X, y = utils.lower_sample_data(X, y)


        data = torch.from_numpy(np.array(X).astype(float)).float()
        label = np.array(y)
        print("输入的数据长度：X:",data.size(), "y:", label.shape)

        # 加载训练过的模型
        model = Dnn(2)
        state_dict = torch.load("checkpoints/wavelet_model.pt")
        model.load_state_dict(state_dict)
        model.eval()
        pre_y = []
        real_y = []
        classIndex = 0
        # len(data) - 128
        for i in tqdm(range(len(label)-cfg.batch_size)):
            with torch.no_grad():
                data_tensor = data[i:i+cfg.batch_size]
                tu = model(data_tensor)
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


