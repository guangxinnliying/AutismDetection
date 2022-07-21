# -*- coding: utf-8 -*-

import torch.nn as nn
 
class MobileNetV1(nn.Module):
    def __init__(self):
        super(MobileNetV1, self).__init__()
 
        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )
 
        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
    
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )
 
        self.model = nn.Sequential(
            conv_bn(  3,  32, 2), 
            conv_dw( 32,  64, 1),
            conv_dw( 64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
            nn.AvgPool2d(2) #44*44图像最后卷积得到2*2*1024矩阵，所以要用2来池化，变成1*1*1024
        )
        self.fc = nn.Linear(1024, 7)

 
    def forward(self, x):
        x = self.model(x)  #这一句出错
        print("x.size(0):",x.size(0)," x.shape:",x.shape)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x