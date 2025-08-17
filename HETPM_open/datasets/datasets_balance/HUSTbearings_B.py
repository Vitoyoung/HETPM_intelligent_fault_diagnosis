from numpy import loadtxt
from scipy.io import loadmat
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets.SequenceDatasets import dataset
from datasets.sequence_aug import *
from tqdm import tqdm
from random import sample
from tqdm import tqdm
from scipy.stats import wasserstein_distance as wd
import matplotlib.pyplot as plt
import csv


signal_size = 1024
moving_step = 1024
sample_number = 100


dataname = {0: ["B_70Hz.csv",  "I_70Hz.csv",  "O_70Hz.csv",  "H_70Hz.csv", "C_70Hz.csv"],           #  70Hz
            1: ["B_75Hz.csv",  "I_75Hz.csv",  "O_75Hz.csv",  "H_75Hz.csv", "C_75Hz.csv"],           #  75Hz
            2: ["B_80Hz.csv",  "I_80Hz.csv",  "O_80Hz.csv",  "H_80Hz.csv", "C_80Hz.csv"],}          #  80Hz


label = [i for i in range(0, 5)]


def get_files(root, N):
    data = []
    lab =[]
    for k in range(len(N)):
        for i in range(len(dataname[N[k]])):
            data_avg = []
            label_avg = []
            path1 = os.path.join('/tmp', root, dataname[N[k]][i])
            data1, lab1 = data_load(path1, label=label[i])

            list_index = [i for i in range(len(data1))]
            index_sample = sample(list_index, sample_number)
            for j in range(len(index_sample)):
                data_avg.append(data1[j])
                label_avg.append(lab1[j])
            data += data_avg
            lab += label_avg


    return [data, lab]

def data_load(filename, label):
    '''
    This function is mainly used to generate test data and training data.
    filename:Data location
    '''

    reader = pd.read_csv(filename, skiprows=22, sep='\t')  # 创建 读取器
    ls = reader.iloc[:, -3:-2]
    ls = np.array(ls)

    data = []
    lab = []
    start, end = 0, signal_size
    while end <= ls.shape[0]:
        data.append(ls[start:end])
        lab.append(label)
        start += moving_step
        end += moving_step

    return data, lab


#--------------------------------------------------------------------------------------------------------------------
class HUST_bearingsB(object):
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

            print('source_train', len(source_train), end='  ')
            print('source_val', len(source_val))
            print('target_train', len(target_train), end='  ')
            print('target_val', len(target_val))

            return source_train, source_val, target_train, target_val


        else:
            #get source train and val
            list_data = get_files(self.data_dir, self.source_N)
            data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
            train_pd, val_pd = train_test_split(data_pd, test_size=0.1, random_state=40, stratify=data_pd["label"])

            source_train = dataset(list_data=train_pd, transform=self.data_transforms['train'])
            source_val = dataset(list_data=val_pd, transform=self.data_transforms['val'])

            # get target train and val
            list_data = get_files(self.data_dir, self.target_N)
            data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
            target_val = dataset(list_data=data_pd, transform=self.data_transforms['val'])
            return source_train, source_val, target_val

