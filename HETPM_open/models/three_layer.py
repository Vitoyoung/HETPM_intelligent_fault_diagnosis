import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.distributions.normal as normal

class three_layer_classifier(nn.Module):
    def __init__(self, input_hidden,  class_number, hidden1=1024, hidden2=256):

        super(three_layer_classifier, self).__init__()

        self.fc1 = nn.Sequential(
            nn.Linear(input_hidden, hidden1),
            nn.ReLU(inplace=True),
            nn.Dropout()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(inplace=True),
            nn.Dropout()
        )

        self.fc3 = nn.Sequential(
            nn.Linear(hidden2, class_number),
            nn.BatchNorm1d(class_number),
            nn.ReLU(inplace=True),
            nn.Dropout()
        )



    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x