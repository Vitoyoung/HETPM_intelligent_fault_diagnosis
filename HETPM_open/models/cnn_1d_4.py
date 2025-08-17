#!/usr/bin/python
# -*- coding:utf-8 -*-
from torch import nn
import warnings


class CNN_4(nn.Module):
    def __init__(self, in_channel=1):
        super(CNN_4, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv1d(in_channel, 16, kernel_size=32, stride=1, padding=16),
            nn.BatchNorm1d(16),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=16, stride=1, padding=8),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )
        self.block3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )
        self.block4 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        #x = x.view(x.size(0), 1, -1)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = x.view(x.size(0), -1) #64*64=4096*1
        x = self.dropout(x)

        return x


# convnet without the last layer
class cnn_features_4(nn.Module):
    def __init__(self):
        super(cnn_features_4, self).__init__()
        self.model_cnn = CNN_4()
        self.__in_features = 4096

    def forward(self, x):
        x = self.model_cnn(x)
        return x

    def output_num(self):
        return self.__in_features
