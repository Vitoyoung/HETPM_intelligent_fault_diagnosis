import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets.SequenceDatasets import dataset
from datasets.sequence_aug import *
from tqdm import tqdm
from random import sample

signal_size = 1024
moving_step = 298
sample_number = 400

# Three working conditions
dataname = {0: ["ib600_2.csv", "n600_3_2.csv", "ob600_2.csv", "tb600_2.csv"],
            1: ["ib800_2.csv", "n800_3_2.csv", "ob800_2.csv", "tb800_2.csv"],
            2: ["ib1000_2.csv", "n1000_3_2.csv", "ob1000_2.csv", "tb1000_2.csv"]}

label = [i for i in range(0, 4)]


# generate Training Dataset and Testing Dataset
def get_files(root, N):
    '''
    This function is used to generate the final training set and test set.
    root:The location of the data set
    '''

    data = []
    lab = []
    for k in range(len(N)):
        for i in (range(len(dataname[N[k]]))):
            data_avg = []
            label_avg = []
            path1 = os.path.join('/tmp', root, dataname[N[k]][i])
            data1, lab1 = data_load(path1, label=label[i])
            #-------------------------------------------------------------------------------------------------------------
            list_index = [i for i in range(len(data1))]
            index_sample = sample(list_index,sample_number)
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
    fl = np.loadtxt(filename)
    fl = fl.reshape(-1, 1)
    data = []
    lab = []
    start, end = 0, signal_size
    while end <= fl.shape[0]:
        data.append(fl[start:end])
        lab.append(label)
        start += moving_step
        end += moving_step

    return data, lab


# --------------------------------------------------------------------------------------------------------------------
class JNUB(object):
    num_classes = 4
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
            train_pd, val_pd = train_test_split(data_pd, test_size=0.2, random_state=40, stratify=data_pd["label"])
            source_train = dataset(list_data=train_pd, transform=self.data_transforms['train'])
            source_val = dataset(list_data=val_pd, transform=self.data_transforms['val'])

            # get target train and val
            list_data = get_files(self.data_dir, self.target_N)
            data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
            train_pd, val_pd = train_test_split(data_pd, test_size=0.2, random_state=40, stratify=data_pd["label"])
            target_train = dataset(list_data=train_pd, transform=self.data_transforms['train'])
            target_val = dataset(list_data=val_pd, transform=self.data_transforms['val'])
            # --------------------------------------------------------------------------------
            print('source_train', len(source_train), end='  ')
            print('source_val', len(source_val))
            print('target_train', len(target_train), end='  ')
            print('target_val', len(target_val))

            print('signal_size', signal_size, end='     ')
            print('moving_step', moving_step, end='     ')
            print('sample_number', sample_number)

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



