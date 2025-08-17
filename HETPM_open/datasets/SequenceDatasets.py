#!/usr/bin/python
# -*- coding:utf-8 -*-

import torch
from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms
import numpy as np
from datasets.sequence_aug import *

class dataset(Dataset):

    def __init__(self, list_data, test=False, transform=None):
        self.test = test
        if self.test:
            self.seq_data = list_data['data'].tolist()
        else:
            self.seq_data = list_data['data'].tolist()
            self.labels = list_data['label'].tolist()
        if transform is None:
            self.transforms = Compose([
                Reshape()
            ])
        else:
            self.transforms = transform


    def __len__(self):
        return len(self.seq_data)

    def __getitem__(self, item):
        if self.test:
            seq = self.seq_data[item]
            seq = self.transforms(seq)
            return seq, item
        else:
            seq = self.seq_data[item]
            label = self.labels[item]
            seq = self.transforms(seq)
            return seq, label


# for ECB  format as [data, data_weak, data_strong]
class dataset_ECB(Dataset):

    def __init__(self, list_data, test=False, transform=None):
        self.test = test
        if self.test:
            self.seq_data = list_data['data'].tolist()
            self.seq_data_data_weakNoise = list_data['data_weakNoise'].tolist()
            self.seq_data_data_strongNoise = list_data['data_strongNoise'].tolist()
        else:
            self.seq_data = list_data['data'].tolist()
            self.seq_data_data_weakNoise = list_data['data_weakNoise'].tolist()
            self.seq_data_data_strongNoise = list_data['data_strongNoise'].tolist()
            self.labels = list_data['label'].tolist()
        if transform is None:
            self.transforms = Compose([
                Reshape()
            ])
        else:
            self.transforms = transform


    def __len__(self):
        return len(self.seq_data)


    def __getitem__(self, item):
        if self.test:
            seq = self.seq_data[item]
            seq = self.transforms(seq)

            seq_Weak = self.seq_data_data_weakNoise[item]
            seq_Weak = self.transforms(seq_Weak)

            seq_Strong = self.seq_data_data_strongNoise[item]
            seq_Strong = self.transforms(seq_Strong)

            # print('__getitem__ test')
            return seq, seq_Weak, seq_Strong, item


        else:
            seq = self.seq_data[item]
            seq = self.transforms(seq)

            seq_Weak = self.seq_data_data_weakNoise[item]
            seq_Weak = self.transforms(seq_Weak)

            seq_Strong = self.seq_data_data_strongNoise[item]
            seq_Strong = self.transforms(seq_Strong)

            label = self.labels[item]

            # print('__getitem__ train')
            return seq, seq_Weak, seq_Strong, label

