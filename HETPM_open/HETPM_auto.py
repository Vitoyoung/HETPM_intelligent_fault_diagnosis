# Collaborative and Conditional Deep Adversarial Network for Intelligent Bearing Fault Diagnosis
#!/usr/bin/python
# -*- coding:utf-8 -*-
import argparse
import os
import time
from datetime import datetime
from time import strftime
from time import gmtime
import logging
from utils.train_util_HETPM import train_utils_auto
import torch
import warnings
import numpy as np
from colorama import Fore, Back, Style, init
print("torch.__version__",torch.__version__)
warnings.filterwarnings('ignore')

args = None

def parse_args():
    parser = argparse.ArgumentParser(description='Train')

    # CNN-Vit is G1  CNN is G2
    parser.add_argument('--model_name_G1', type=str, default='VitLiConv_features_4096V1_1d', help='the name of the Vit')
    parser.add_argument('--model_name_G2', type=str, default='CNN_4_1d', help='the name of the CNN')

    # use two_layers        use two_layers_tsne if visualize
    parser.add_argument('--which_classifier', type=str, choices=['one_layers', 'two_layers', 'three_layers', 'two_layers_tsne'],
                        default='two_layers_tsne')


    parser.add_argument('--data_name', type=str, default='PT1TL_B', help='transfer learning dataset')
    # JNUB  HUST_bearingsB
    # PT1TL_B

    parser.add_argument('--repeat_times', type=int, default=2, help='experiment repeat times')
    parser.add_argument('--pic_form_dir', type=str, default='F:\Experiments\\base', help='transfer learning dataset')
    parser.add_argument('--normlizetype', type=str, default='mean-std', help='nomalization type')

    # training parameters
    parser.add_argument('--cuda_device', type=str, default='0', help='assign device')
    parser.add_argument('--batch_size', type=int, default=32, help='batchsize of the training process')
    parser.add_argument('--num_workers', type=int, default=0, help='the number of training process')
    parser.add_argument('--last_batch', type=bool, default=True, help='whether using the last batch')


    # mutual-teaching  threshould of branch
    parser.add_argument('--vit_to_cnn_value', type=float, default=0.99)      # CNN teach CNN-ViT  0.95
    parser.add_argument('--cnn_to_vit_value', type=float, default=0.99)      # CNN-ViT teach CNN  0.99
    parser.add_argument('--adaptive_threshold', type=bool, default=True)


    # use output og F1 or F2 or mean
    parser.add_argument('--predict_by', type=str, choices=['average_predict', 'F2_predict', 'F1_predict'], default='F2_predict')



    # Maximum_Classifier_Discrepancy(MCD) or BiClassifier(BCDM)-------------------------------------------------------
    parser.add_argument('--func', type=str, choices=['L1', 'MSE', 'Cosine', 'SWD', 'CDD'], default='CDD', help='  ')
    parser.add_argument('--num_k', type=int, default=4, help='  ')




    parser.add_argument('--opt', type=str, choices=['sgd', 'adam'], default='adam', help='the optimizer')
    parser.add_argument('--lr_Vit', type=float, default=1e-3, help='the initial learning rate')
    parser.add_argument('--lr_CNN', type=float, default=1e-3, help='the initial learning rate')

    parser.add_argument('--momentum', type=float, default=0.9, help='the momentum for sgd')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='the weight decay')
    parser.add_argument('--lr_scheduler', type=str, choices=['step', 'exp', 'stepLR', 'fix'], default='step', help='the learning rate schedule')
    # Step
    parser.add_argument('--gamma', type=float, default=0.1, help='learning rate scheduler parameter for step and exp')
    parser.add_argument('--steps', type=str, default='50', help='the learning rate decay for step and stepLR')


    parser.add_argument('--max_epoch', type=int, default=100, help='max number of epoch')

    # confusion matrix
    parser.add_argument('--confusion_matrix', type=bool, default=True, help='')
    parser.add_argument('--conf_matrix_normalize', type=bool, default=True, help='')

    # t-SNE
    parser.add_argument('--t_SNE', type=bool, default=True, help='')
    parser.add_argument('--t_SNE_2D', type=bool, default=True, help='')
    parser.add_argument('--t_SNE_3D', type=bool, default=False, help='')
    parser.add_argument('--t_SNE_auto_color', type=bool, default=False, help='')

    # array
    parser.add_argument('--save_array', type=bool, default=False, help='')


    args = parser.parse_args()
    return args



if __name__ == '__main__':

    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    task_start = time.time()
    args = parse_args()

    if "JNU" in args.data_name:
        work_condition_length = 3
    if "PT1TL" in args.data_name:
        work_condition_length = 5
    if "PT2TL" in args.data_name:
        work_condition_length = 5
    if "HUST_bearings" in args.data_name:
        work_condition_length = 3


    for k, v in args.__dict__.items():
        logging.info("{}: {}".format(k, v))

    # print(args)
    for k, v in sorted(vars(args).items()):
        print(k, '=', v)


    print('{}{}'.format("dataset is ", args.data_name))

    task_acc_mean_F1 = []
    task_std_mean_F1 = []

    task_acc_mean_F2 = []
    task_std_mean_F2 = []

    task_acc_mean_Avg = []
    task_std_mean_Avg = []


    task_index = work_condition_length * (work_condition_length-1)

    for i in range(work_condition_length):
        for j in range(work_condition_length):
            # i = 3
            # j = 4
            if i != j:

                # for sensitive in [2, 3, 4, 5, 6, 7, 8, 9, 10]:#
                #     args.num_classifier = sensitive
                #     print("sensitive", sensitive)
                #     print('num_classifier', args.num_classifier)
                # -----------------------------------------------------------------------------------------------
                task_index = task_index - 1
                print('{}{}{}{}{}'.format("------------now task from 【", i, "】to【", j, "】------------"))

                last_acc_F1_for_pic = []
                last_acc_F2_for_pic = []
                last_acc_Avg_for_pic = []

                for r in range(args.repeat_times):
                    time_rest_start = time.time()
                    print(r+1, "/", args.repeat_times)
                    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device.strip()


                    trainer = train_utils_auto(args)
                    trainer.setup([[i], [j]])
                    last_acc_F1, last_acc_F2, last_acc_Avg = trainer.train([[i],[j]], r)

                    last_acc_F1_for_pic.append(last_acc_F1)
                    last_acc_F2_for_pic.append(last_acc_F2)
                    last_acc_Avg_for_pic.append(last_acc_Avg)

                    time_rest_end = time.time() - time_rest_start
                    time_rest_all = time_rest_end * (task_index * args.repeat_times + (args.repeat_times-r-1))
                    time_rest_all = strftime("%H:%M:%S", gmtime(time_rest_all))
                    print("left", time_rest_all)

                last_acc_F1_for_pic = np.array(last_acc_F1_for_pic)
                acc_mean_F1 = last_acc_F1_for_pic.mean()
                acc_std_F1 = last_acc_F1_for_pic.std()

                last_acc_F2_for_pic = np.array(last_acc_F2_for_pic)
                acc_mean_F2 = last_acc_F2_for_pic.mean()
                acc_std_F2 = last_acc_F2_for_pic.std()

                last_acc_Avg_for_pic = np.array(last_acc_Avg_for_pic)
                acc_mean_Avg = last_acc_Avg_for_pic.mean()
                acc_std_Avg = last_acc_Avg_for_pic.std()

                print('acc_mean_F1', acc_mean_F1, 'acc_std_F1', acc_std_F1)
                print(Fore.RED + '{}{}{}{}{}{}{}{}'.format("-transfer task【", i, "】-【", j, "】 mean acc of F1 is ", round(acc_mean_F1, 2), "，std is ", round(acc_std_F1, 2)))
                print(Style.RESET_ALL)

                print('acc_mean_F2', acc_mean_F2, 'acc_std_F2', acc_std_F2)
                print(Fore.RED + '{}{}{}{}{}{}{}{}'.format("-transfer task【", i, "】-【", j, "】mean acc of F2 is ", round(acc_mean_F2, 2), "，std is ", round(acc_std_F2, 2)))
                print(Style.RESET_ALL)

                print('acc_mean_Avg', acc_mean_Avg, 'acc_std_Avg', acc_std_Avg)
                print(Fore.RED + '{}{}{}{}{}{}{}{}'.format("-transfer task【", i, "】-【", j, "】mean acc of F1+F2 is ", round(acc_mean_Avg, 2), "，std is ", round(acc_std_Avg, 2)))
                print(Style.RESET_ALL)

                task_acc_mean_F1.append(acc_mean_F1)
                task_std_mean_F1.append(acc_std_F1)

                task_acc_mean_F2.append(acc_mean_F2)
                task_std_mean_F2.append(acc_std_F2)

                task_acc_mean_Avg.append(acc_mean_Avg)
                task_std_mean_Avg.append(acc_std_Avg)


                # -----------------------------------------------------------------------------------------------

    task_acc_mean_F1 = np.array(task_acc_mean_F1)
    task_std_mean_F1 = np.array(task_std_mean_F1)
    task_acc_mean_F1 = task_acc_mean_F1.mean()
    task_std_mean_F1 = task_std_mean_F1.mean()

    task_acc_mean_F2 = np.array(task_acc_mean_F2)
    task_std_mean_F2 = np.array(task_std_mean_F2)
    task_acc_mean_F2 = task_acc_mean_F2.mean()
    task_std_mean_F2 = task_std_mean_F2.mean()

    task_acc_mean_Avg = np.array(task_acc_mean_Avg)
    task_std_mean_Avg = np.array(task_std_mean_Avg)
    task_acc_mean_Avg = task_acc_mean_Avg.mean()
    task_std_mean_Avg = task_std_mean_Avg.mean()


    print("     ")

    print('task_acc_mean_F1', task_acc_mean_F1, 'task_std_mean_F1', task_std_mean_F1)
    print(Fore.RED + '{}{}{}{}'.format("-acc of F1 on dataset is ", round(task_acc_mean_F1, 2), "，std is ", round(task_std_mean_F1, 2)))
    print(Style.RESET_ALL)

    print('task_acc_mean_F2', task_acc_mean_F2, 'task_std_mean_F2', task_std_mean_F2)
    print(Fore.RED + '{}{}{}{}'.format("-acc of F2 on dataset is ", round(task_acc_mean_F2, 2), "，std is ", round(task_std_mean_F2, 2)))
    print(Style.RESET_ALL)

    print('task_acc_mean_Avg', task_acc_mean_Avg, 'task_std_mean_Avg', task_std_mean_Avg)
    print(Fore.RED + '{}{}{}{}'.format("-acc of F1+F2 on dataset is ", round(task_acc_mean_Avg, 2), "，std is ", round(task_std_mean_Avg, 2)))
    print(Style.RESET_ALL)
    print("time：", strftime("%H:%M:%S", gmtime(time.time() - task_start)))
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))