import math
import torch.nn as nn
from layers import BBB_Linear, BBB_Conv2d
from layers import BBB_LRT_Linear, BBB_LRT_Conv2d
from layers import FlattenLayer, ModuleWrapper


class BBBLeNet(ModuleWrapper):
    '''The architecture of LeNet with Bayesian Layers'''

    def __init__(self, outputs, inputs, priors, layer_type='lrt', activation_type='softplus'):
        super(BBBLeNet, self).__init__()
        self.num_classes = outputs
        self.layer_type = layer_type
        self.priors = priors

        if layer_type=='lrt':
            BBBLinear = BBB_LRT_Linear
            BBBConv2d = BBB_LRT_Conv2d
        elif layer_type=='bbb':
            BBBLinear = BBB_Linear
            BBBConv2d = BBB_Conv2d
        else:
            raise ValueError("Undefined layer_type")
        
        if activation_type=='softplus':
            self.act = nn.Softplus
        elif activation_type=='relu':
            self.act = nn.ReLU
        else:
            raise ValueError("Only softplus or relu supported")

        self.conv1 = BBBConv2d(inputs, 32, 4, padding=0, bias=True, priors=self.priors)
        self.act1 = self.act()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=1)

        self.conv2 = BBBConv2d(32, 64, 2, padding=0, bias=True, priors=self.priors)
        self.act2 = self.act()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=1)

        length = 2* 26 * 64
        self.flatten = FlattenLayer(length)
        self.fc1 = BBBLinear(length, 128, bias=True, priors=self.priors)
        self.act3 = self.act()

        self.fc2 = BBBLinear(128, 64, bias=True, priors=self.priors)
        self.act4 = self.act()

        self.fc3 = BBBLinear(64, outputs, bias=True, priors=self.priors)