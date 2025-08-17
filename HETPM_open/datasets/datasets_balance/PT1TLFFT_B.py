import os
from scipy.io import loadmat
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets.SequenceDatasets import dataset
from datasets.sequence_aug import *
from tqdm import tqdm
from random import sample


signal_size = 1024
moving_step = 100
sample_number = 100

# Three working conditions
dataname = {0: ["N_800.mat",  "B_800.mat",  "C_800.mat",  "I_800.mat",  "O_800.mat"],       #  800
            1: ["N_1200.mat", "B_1200.mat", "C_1200.mat", "I_1200.mat", "O_1200.mat"],      # 1200
            2: ["N_1600.mat", "B_1600.mat", "C_1600.mat", "I_1600.mat", "O_1600.mat"],      # 1600
            3: ["N_2000.mat", "B_2000.mat", "C_2000.mat", "I_2000.mat", "O_2000.mat"],      # 2000
            4: ["N_2400.mat", "B_2400.mat", "C_2400.mat", "I_2400.mat", "O_2400.mat"]}      # 2400


label = [i for i in range(0, 5)]


# generate Training Dataset and Testing Dataset
def get_files(root, N):
    '''
    This function is used to generate the final training set and test set.
    root:The location of the data set
    '''

    data = []
    lab = []
    for k in range(len(N)):
        for i in range(len(dataname[N[k]])):
            data_avg = []
            label_avg = []
            path1 = os.path.join('/tmp', root, dataname[N[k]][i])
            data1, lab1 = data_load(path1, label=label[i])
            #-------------------------------------------------------------------------------------------------------------
            list_index = [i for i in range(len(data1))]
            index_sample = sample(list_index, sample_number)
            for j in range(len(index_sample)):
                data_avg.append(data1[j])
                label_avg.append(lab1[j])
            data += data_avg
            lab += label_avg
            #--------------------------------------------------------------------------------------------------------------


    return [data, lab]


def data_load(filename, label):
    '''
    This function is mainly used to generate test data and training data.
    filename:Data location
    '''
    fl = loadmat(filename)
    fl = fl['Data0']  #
    data = []
    lab = []
    start, end = 0, signal_size
    while end <= fl.shape[0]:
        x = fl[start:end]
        x = np.fft.fft(x)
        x = np.abs(x) / len(x)
        x = x[range(int(x.shape[0] / 2))]
        x = x.reshape(-1, 1)
        data.append(x)
        lab.append(label)
        start += moving_step
        end += moving_step

    return data, lab


# --------------------------------------------------------------------------------------------------------------------
class PT1TLFFT_B(object):
    num_classes = 5
    inputchannel = 1

    def __init__(self, data_dir, transfer_task, normlizetype="0-1"):
        self.data_dir = data_dir
        self.source_N = transfer_task[0]
        self.target_N = transfer_task[1]
        self.normlizetype = normlizetype
        self.data_transforms = {
            'train': Compose([
                Reshape(),
                Normalize(self.normlizetype),
                # RandomAddGaussian(),
                # RandomScale(),
                # RandomStretch(),
                # RandomCrop(),
                Retype(),
                # Scale(1)
            ]),
            'val': Compose([
                Reshape(),
                Normalize(self.normlizetype),
                Retype(),
                # Scale(1)
            ])
        }

    def data_split(self, transfer_learning=True):
        if transfer_learning:
            # get source train and val
            list_data = get_files(self.data_dir, self.source_N)
            data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
            train_pd, val_pd = train_test_split(data_pd, test_size=0.5, random_state=40, stratify=data_pd["label"])
            source_train = dataset(list_data=train_pd, transform=self.data_transforms['train'])
            source_val = dataset(list_data=val_pd, transform=self.data_transforms['val'])

            # get target train and val
            list_data = get_files(self.data_dir, self.target_N)
            data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
            train_pd, val_pd = train_test_split(data_pd, test_size=0.5, random_state=40, stratify=data_pd["label"])
            target_train = dataset(list_data=train_pd, transform=self.data_transforms['train'])
            target_val = dataset(list_data=val_pd, transform=self.data_transforms['val'])
            # --------------------------------------------------------------------------------
            print('source_train', len(source_train), end='  ')
            print('source_val', len(source_val))
            print('target_train', len(target_train), end='  ')
            print('target_val', len(target_val))
            return source_train, source_val, target_train, target_val
        else:
            # get source train and val
            list_data = get_files(self.data_dir, self.source_N)
            data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
            train_pd, val_pd = train_test_split(data_pd, test_size=0.2, random_state=40, stratify=data_pd["label"])
            source_train = dataset(list_data=train_pd, transform=self.data_transforms['train'])
            source_val = dataset(list_data=val_pd, transform=self.data_transforms['val'])

            # get target train and val
            list_data = get_files(self.data_dir, self.target_N)
            data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
            target_val = dataset(list_data=data_pd, transform=self.data_transforms['val'])
            return source_train, source_val, target_val



