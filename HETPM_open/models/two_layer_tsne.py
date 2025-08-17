import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.distributions.normal as normal

class two_layer_classifier_tsne(nn.Module):
    def __init__(self,
                input_hidden,
                class_number,
                hidden=128
                ):

        super(two_layer_classifier_tsne, self).__init__()

        self.fc1 = nn.Linear(input_hidden, hidden)
        self.bn1_fc = nn.BatchNorm1d(hidden)

        self.fc2 = nn.Linear(hidden, class_number)





    def forward(self, x):
        x = self.fc1(x)
        x_tsne = F.relu(self.bn1_fc(x))

        x = self.fc2(x_tsne)


        return x, x_tsne