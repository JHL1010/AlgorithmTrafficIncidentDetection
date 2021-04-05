import torch
import torch.nn as nn

import numpy as np

from model.wavlet import Wavelet

class Dnn(nn.Module):

    def __init__(self, outputs):
        super().__init__()
        self.num_classes = outputs
        self.act = nn.ReLU()
        self.inputs = 8
        #  layer
        self.layer1 = nn.Linear(self.inputs, 32, bias=True)
        self.layer2 = nn.Linear(32, 16, bias=True)
        self.layer3 = nn.Linear(16, outputs, bias=True)

    def forward(self, x):
        self.inputs, x = Wavelet(x)
        out = self.layer1(x)
        out = self.act(out)
        out = self.layer2(out)
        out = self.act(out)
        out = self.layer3(out)
        return out
