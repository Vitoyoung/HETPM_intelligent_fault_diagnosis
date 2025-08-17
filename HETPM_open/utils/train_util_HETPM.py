import warnings

import torch, gc
from torch import nn
from torch import optim
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import time
from sklearn.manifold import TSNE
import itertools
from mpl_toolkits.mplot3d import Axes3D
from colorama import Fore, Style
from tqdm import tqdm
import math
#-----------------------------------------------
import datasets
import models

from utils.loss import consistency_loss


from utils.different_metric import discrepancy
from utils.different_metric import discrepancy_mse
from utils.different_metric import discrepancy_cos
from utils.different_metric import discrepancy_slice_wasserstein
from utils.different_metric import cdd


class train_utils_auto(object):

    def __init__(self, args):
        self.args = args


    #初始化
    def setup(self, auto_transfer_task):

        args = self.args

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            assert args.batch_size % self.device_count == 0, "batch size should be divided by device count"

        else:
            print("gpu is not available")
            self.device = torch.device("cpu")
            self.device_count = 1
            # logging.info('using {} cpu'.format(self.device_count))

        # self.device = torch.device("cpu")

        Dataset = getattr(datasets, args.data_name)
        self.datasets = {}

        if isinstance(auto_transfer_task[0], str):
           print("args.transfer_task", auto_transfer_task)
           auto_transfer_task = eval("".join(auto_transfer_task))

        global transfer_task
        transfer_task = auto_transfer_task

        if "JNU" in args.data_name:
            data_direction = 'F:\故障诊断数据集\江南大学轴承数据(一)\数据'
        if "PT1TL" in args.data_name:
            data_direction = 'F:\故障诊断数据集\PTTL\辅助轴承'
        if "PT2TL" in args.data_name:
            data_direction = 'F:\故障诊断数据集\PTTL\故障轴承'
        if "PT1TL_50kHz" in args.data_name:
            data_direction = 'F:\故障诊断数据集\PT1TL50KHZ\辅助轴承'
        if "PT2TL_50kHz" in args.data_name:
            data_direction = 'F:\故障诊断数据集\PT2TL50KHZ\故障轴承'
        if "HUST_bearings" in args.data_name:
            data_direction = 'F:\故障诊断数据集\HUST_bearing_dataset\Raw_data_csv'



        if (len((args.data_name).split('_')) >= 2) and (args.data_name).split('_')[1] == 'ECB':
            self.datasets['source_train'], self.datasets['source_val'], \
                self.datasets['target_train'], self.datasets['target_val'] = Dataset(data_direction, auto_transfer_task,
                                                                                     args.weak_noise, args.strong_noise,
                                                                                     args.normlizetype).data_split(
                transfer_learning=True)
        else:
            self.datasets['source_train'], self.datasets['source_val'], \
                self.datasets['target_train'], self.datasets['target_val'] = Dataset(data_direction, auto_transfer_task,
                                                                                     args.normlizetype).data_split(
                transfer_learning=True)

        global source_train_length
        global source_val_length
        global target_train_length
        global target_val_length
        source_train_length = len(self.datasets['source_train'])
        source_val_length = len(self.datasets['source_val'])
        target_train_length = len(self.datasets['target_train'])
        target_val_length = len(self.datasets['target_val'])


        self.dataloaders = {x: torch.utils.data.DataLoader(self.datasets[x], batch_size=args.batch_size,
                                                           shuffle=(True if x.split('_')[1] == 'train' else False),
                                                           num_workers=args.num_workers,
                                                           pin_memory=(True if self.device == 'cuda' else False),
                                                           drop_last=(True if args.last_batch and x.split('_')[1] == 'train' else False))
                            for x in ['source_train', 'source_val', 'target_train', 'target_val']}



        self.G1 = getattr(models, args.model_name_G1)()
        self.G2 = getattr(models, args.model_name_G2)()


        self.class_dim_1 = self.G1.output_num()
        self.class_dim_2 = self.G2.output_num()
        if self.class_dim_1 == self.class_dim_2:
            self.class_dim = self.class_dim_2
        else:
            print(' Different Output Dimension ')



        if args.which_classifier == "one_layers":
            self.F1 = nn.Linear(self.class_dim, Dataset.num_classes)
            self.F2 = nn.Linear(self.class_dim, Dataset.num_classes)

        elif args.which_classifier == 'two_layers':
            self.F1 = getattr(models, 'two_layers')(self.class_dim, Dataset.num_classes)
            self.F2 = getattr(models, 'two_layers')(self.class_dim, Dataset.num_classes)

        elif args.which_classifier == "two_layers_tsne":
            self.F1 = getattr(models, 'two_layers_tsne')(self.class_dim, Dataset.num_classes)
            self.F2 = getattr(models, 'two_layers_tsne')(self.class_dim, Dataset.num_classes)

        elif args.which_classifier == 'three_layers':
            self.F1 = getattr(models, 'three_layers')(input_hidden=self.class_dim, hidden1=1024,
                                                      hidden2=256, class_number=Dataset.num_classes)
            self.F2 = getattr(models, 'three_layers')(input_hidden=self.class_dim, hidden1=1024,
                                                      hidden2=256, class_number=Dataset.num_classes)
        else:
            print('classifier is not choosen')





        G1_parameters = [{"params": self.G1.parameters(), "lr": args.lr_Vit}]
        G2_parameters = [{"params": self.G2.parameters(), "lr": args.lr_CNN}]

        F1_parameters = [{"params": self.F1.parameters(), "lr": args.lr_Vit}]
        F2_parameters = [{"params": self.F2.parameters(), "lr": args.lr_CNN}]







        if args.opt == 'sgd':
            self.optimizer_G1 = optim.SGD(G1_parameters, lr=args.lr_Vit, momentum=args.momentum,
                                          weight_decay=args.weight_decay)
            self.optimizer_G2 = optim.SGD(G2_parameters, lr=args.lr_CNN, momentum=args.momentum,
                                          weight_decay=args.weight_decay)

            self.optimizer_F1 = optim.SGD(F1_parameters, lr=args.lr_Vit, momentum=args.momentum,
                                          weight_decay=args.weight_decay)
            self.optimizer_F2 = optim.SGD(F2_parameters, lr=args.lr_CNN, momentum=args.momentum,
                                          weight_decay=args.weight_decay)


        elif args.opt == 'adam':
            self.optimizer_G1 = optim.Adam(G1_parameters, lr=args.lr_Vit, weight_decay=args.weight_decay)
            self.optimizer_G2 = optim.Adam(G2_parameters, lr=args.lr_CNN, weight_decay=args.weight_decay)

            self.optimizer_F1 = optim.Adam(F1_parameters, lr=args.lr_Vit, weight_decay=args.weight_decay)
            self.optimizer_F2 = optim.Adam(F2_parameters, lr=args.lr_CNN, weight_decay=args.weight_decay)


        else:
            raise Exception("optimizer error")




        if args.lr_scheduler == 'step':
            steps = [int(step) for step in args.steps.split(',')]
            self.lr_scheduler_G1 = optim.lr_scheduler.MultiStepLR(self.optimizer_G1, steps, gamma=args.gamma)
            self.lr_scheduler_F1 = optim.lr_scheduler.MultiStepLR(self.optimizer_F1, steps, gamma=args.gamma)
            self.lr_scheduler_G2 = optim.lr_scheduler.MultiStepLR(self.optimizer_G2, steps, gamma=args.gamma)
            self.lr_scheduler_F2 = optim.lr_scheduler.MultiStepLR(self.optimizer_F2, steps, gamma=args.gamma)

        elif args.lr_scheduler == 'exp':
            self.lr_scheduler_G1 = optim.lr_scheduler.ExponentialLR(self.optimizer_G1, args.gamma)
            self.lr_scheduler_F1 = optim.lr_scheduler.ExponentialLR(self.optimizer_F1, args.gamma)
            self.lr_scheduler_G2 = optim.lr_scheduler.ExponentialLR(self.optimizer_G2, args.gamma)
            self.lr_scheduler_F2 = optim.lr_scheduler.ExponentialLR(self.optimizer_F2, args.gamma)

        elif args.lr_scheduler == 'stepLR':
            steps = int(args.steps)
            self.lr_scheduler_G1 = optim.lr_scheduler.StepLR(self.optimizer_G1, steps, args.gamma)
            self.lr_scheduler_F1 = optim.lr_scheduler.StepLR(self.optimizer_F1, steps, args.gamma)
            self.lr_scheduler_G2 = optim.lr_scheduler.StepLR(self.optimizer_G2, steps, args.gamma)
            self.lr_scheduler_F2 = optim.lr_scheduler.StepLR(self.optimizer_F2, steps, args.gamma)

        elif args.lr_scheduler == 'fix':
            self.lr_scheduler_G1 = None
            self.lr_scheduler_F1 = None
            self.lr_scheduler_G2 = None
            self.lr_scheduler_F2 = None

        else:
            raise Exception("lr schedule error")



        self.start_epoch = 0


        self.dis_dict = {'L1': discrepancy,
                         'MSE': discrepancy_mse,
                         'Cosine': discrepancy_cos,
                         'SWD': discrepancy_slice_wasserstein,
                         'CDD': cdd
                         }


        self.G1.to(self.device)
        self.G2.to(self.device)
        self.F1.to(self.device)
        self.F2.to(self.device)



        self.ce_criterion = nn.CrossEntropyLoss()





    #
    def train(self, auto_transfer_task, time_repeat):

        args = self.args
        start_time = time.time()
        step = 0

        batch_count = 0
        batch_loss = 0.0
        batch_acc = 0

        tsne_i_source = 0
        tsne_i_target = 0
        source_train_loss_pic = []
        source_val_loss_pic = []
        target_val_loss_pic = []
        source_train_acc_pic = []
        source_val_acc_pic = []
        target_val_acc_pic = []
        x_for_pic = np.arange(args.max_epoch)
        Dataset = getattr(datasets, args.data_name)
        confusion_matrix_F1 = np.zeros((Dataset.num_classes, Dataset.num_classes))
        confusion_matrix_F2 = np.zeros((Dataset.num_classes, Dataset.num_classes))
        confusion_matrix_Avg = np.zeros((Dataset.num_classes, Dataset.num_classes))



        for epoch in tqdm(range(self.start_epoch, args.max_epoch), position=0):

            iter_target = iter(self.dataloaders['target_train'])
            len_target_loader = len(self.dataloaders['target_train'])


            for phase in ['source_train', 'source_val', 'target_val']:

                epoch_acc = 0.0
                epoch_loss = 0.0

                epoch_acc_F1 = 0
                epoch_acc_F2 = 0
                epoch_acc_Avg = 0


                epoch_length = 0



                if phase == 'source_train':

                    for batch_idx, (inputs, labels) in enumerate(self.dataloaders[phase]):



                        self.F1.train()
                        self.F2.train()
                        self.G1.train()
                        self.G2.train()

                        source_inputs = inputs
                        target_inputs = next(iter_target)
                        target_inputs = target_inputs[0]

                        inputs = torch.cat((source_inputs, target_inputs), dim=0)
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)

                        if (step + 1) % len_target_loader == 0:
                            iter_target = iter(self.dataloaders['target_train'])



                        ##### INITIALIZATION #####

                        ###### CNN-ViT ######
                        self.optimizer_G1.zero_grad()
                        self.optimizer_F1.zero_grad()
                        self.optimizer_G2.zero_grad()
                        self.optimizer_F2.zero_grad()

                        if args.which_classifier == 'two_layers_tsne':
                            vit_logits, _ = self.F1(self.G1(inputs[:labels.size(0), :]))
                        else:
                            vit_logits = self.F1(self.G1(inputs[:labels.size(0), :]))

                        all_loss = self.ce_criterion(vit_logits, labels)

                        all_loss.backward()
                        self.optimizer_G1.step()
                        self.optimizer_F1.step()

                        self.optimizer_G1.zero_grad()
                        self.optimizer_F1.zero_grad()
                        self.optimizer_G2.zero_grad()
                        self.optimizer_F2.zero_grad()



                        ###### CNN ######
                        if args.which_classifier == 'two_layers_tsne':
                            # features = self.G2(inputs[:labels.size(0), :])
                            features = self.G2(inputs)

                            cnn_logits, _ = self.F2(features)

                        else:
                            # features = self.G2(inputs[:labels.size(0), :])

                            features = self.G2(inputs)
                            cnn_logits = self.F2(features)

                        cnn_loss = self.ce_criterion(cnn_logits[:labels.size(0), :], labels)

                        total_loss = cnn_loss

                        total_loss.backward()
                        self.optimizer_G2.step()
                        self.optimizer_F2.step()

                        self.optimizer_G1.zero_grad()
                        self.optimizer_F1.zero_grad()
                        self.optimizer_G2.zero_grad()
                        self.optimizer_F2.zero_grad()



                        ######## biclassifier training ########

                        ##### max discrepancy #####
                        if args.which_classifier == 'two_layers_tsne':
                            output_f1, _ = self.F1(self.G1(inputs[:labels.size(0), :]))
                            output_f2, _ = self.F2(self.G2(inputs[:labels.size(0), :]))
                        else:
                            output_f1 = self.F1(self.G1(inputs[:labels.size(0), :]))
                            output_f2 = self.F2(self.G2(inputs[:labels.size(0), :]))

                        loss_f1 = self.ce_criterion(output_f1, labels)
                        loss_f2 = self.ce_criterion(output_f2, labels)
                        loss_s = loss_f1 + loss_f2

                        vit_features_t = self.G1(inputs[labels.size(0):, :])

                        if args.which_classifier == 'two_layers_tsne':
                            output_t1, _ = self.F1(vit_features_t)
                            output_t2, _ = self.F2(vit_features_t)
                        else:
                            output_t1 = self.F1(vit_features_t)
                            output_t2 = self.F2(vit_features_t)


                        loss_dis = self.dis_dict[args.func](output_t1,output_t2)


                        loss = loss_s - loss_dis

                        loss.backward()
                        self.optimizer_F1.step()
                        self.optimizer_F2.step()

                        self.optimizer_G1.zero_grad()
                        self.optimizer_F1.zero_grad()
                        self.optimizer_G2.zero_grad()
                        self.optimizer_F2.zero_grad()



                        ##### min discrepancy #####
                        for j in range(args.num_k):
                            feat_t = self.G2(inputs[labels.size(0):, :])
                            if args.which_classifier == 'two_layers_tsne':
                                output_t1, _ = self.F1(feat_t)
                                output_t2, _ = self.F2(feat_t)
                            else:
                                output_t1 = self.F1(feat_t)
                                output_t2 = self.F2(feat_t)


                            loss_dis = self.dis_dict[args.func](output_t1, output_t2)

                            loss_dis.backward()
                            self.optimizer_G2.step()

                            self.optimizer_G1.zero_grad()
                            self.optimizer_F1.zero_grad()
                            self.optimizer_G2.zero_grad()
                            self.optimizer_F2.zero_grad()



                        ##### mutual learning #####

                        now_epoch_ratio = epoch / args.max_epoch


                        ####### CNN-ViT to CNN #######
                        if args.which_classifier == 'two_layers_tsne':
                            vit_logits, _ = self.F1(self.G1(inputs[labels.size(0):, :]))
                            logits_u_s, _ = self.F2(self.G2(inputs[labels.size(0):, :]))
                        else:
                            vit_logits = self.F1(self.G1(inputs[labels.size(0):, :]))
                            logits_u_s = self.F2(self.G2(inputs[labels.size(0):, :]))

                        loss_vit_to_cnn = consistency_loss(logits_u_s, vit_logits, threshold=args.vit_to_cnn_value,
                                                           adaptive_threshold=args.adaptive_threshold, epoch_ratio=(1-now_epoch_ratio))

                        loss_vit_to_cnn.backward()
                        self.optimizer_G2.step()
                        self.optimizer_F2.step()

                        self.optimizer_G1.zero_grad()
                        self.optimizer_F1.zero_grad()
                        self.optimizer_G2.zero_grad()
                        self.optimizer_F2.zero_grad()



                        ####### CNN to CNN-ViT #######

                        if args.which_classifier == 'two_layers_tsne':
                            cnn_logits, _ = self.F2(self.G2(inputs[labels.size(0):, :]))
                            logits_u_s, _ = self.F1(self.G1(inputs[labels.size(0):, :]))
                        else:
                            cnn_logits = self.F2(self.G2(inputs[labels.size(0):, :]))
                            logits_u_s = self.F1(self.G1(inputs[labels.size(0):, :]))

                        loss_cnn_to_vit = consistency_loss(logits_u_s, cnn_logits, threshold=args.cnn_to_vit_value,
                                                           adaptive_threshold=args.adaptive_threshold, epoch_ratio=now_epoch_ratio)

                        loss_cnn_to_vit.backward()
                        self.optimizer_G1.step()
                        self.optimizer_F1.step()

                        self.optimizer_G1.zero_grad()
                        self.optimizer_F1.zero_grad()
                        self.optimizer_G2.zero_grad()
                        self.optimizer_F2.zero_grad()


                        ###########################

                        # predict result
                        if args.which_classifier == 'two_layers_tsne':
                            logits_1, _ = self.F1(self.G1(inputs[:labels.size(0), :]))
                            logits_2, _ = self.F2(self.G2(inputs[:labels.size(0), :]))
                        else:
                            logits_1 = self.F1(self.G1(inputs[:labels.size(0), :]))
                            logits_2 = self.F2(self.G2(inputs[:labels.size(0), :]))

                        if args.predict_by == 'average_predict':
                            logits = logits_1 + logits_2
                        elif args.predict_by == 'F2_predict':
                            logits = logits_2

                        pred = logits.argmax(dim=1)

                        correct = torch.eq(pred, labels).float().sum().item()
                        loss_temp = (all_loss).item() * labels.size(0)
                        epoch_loss += loss_temp
                        epoch_acc += correct
                        epoch_length += labels.size(0)

                        batch_loss += loss_temp
                        batch_acc += correct
                        batch_count += labels.size(0)

                        step += 1



                # eval
                else:
                    self.G1.eval()
                    self.G2.eval()
                    self.F1.eval()
                    self.F2.eval()

                    for batch_idx, (inputs, labels) in enumerate(self.dataloaders[phase]):


                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)

                        if (step + 1) % len_target_loader == 0:
                            iter_target = iter(self.dataloaders['target_train'])

                        with torch.set_grad_enabled(False):

                            if args.which_classifier == 'two_layers_tsne':
                                features_1 = self.G1(inputs)
                                outputs_1, features_two_layers_tsne_G1 = self.F1(features_1)
                                features_2 = self.G2(inputs)
                                outputs_2, features_two_layers_tsne_G2 = self.F2(features_2)

                            else:


                                features_2 = self.G2(inputs)
                                outputs_2 = self.F2(features_2)


                                features_1 = self.G1(inputs)
                                outputs_1 = self.F1(features_1)


                            # if args.predict_by == 'average_predict':
                            #     logits = outputs_1 + outputs_2
                            # elif args.predict_by == 'F2_predict':
                            #     logits = outputs_2
                            # elif args.predict_by == 'F1_predict':
                            #     logits = outputs_1


                            logits1 = outputs_1
                            logits2 = outputs_2
                            logits_avg = outputs_1 + outputs_2


                            pred_F1 = logits1.argmax(dim=1)
                            pred_F2 = logits2.argmax(dim=1)
                            pred_avg = logits_avg.argmax(dim=1)

                            # CNN
                            loss = self.ce_criterion(logits2, labels)

                        # plot
                        correct = torch.eq(pred_F2, labels).float().sum().item()

                        correct1 = torch.eq(pred_F1, labels).float().sum().item()
                        correct2 = torch.eq(pred_F2, labels).float().sum().item()
                        correct_avg = torch.eq(pred_avg, labels).float().sum().item()


                        loss_temp = loss.item() * labels.size(0)
                        epoch_loss += loss_temp
                        epoch_acc += correct
                        epoch_length += labels.size(0)

                        epoch_acc_F1 += correct1
                        epoch_acc_F2 += correct2
                        epoch_acc_Avg += correct_avg


                        # t-SNE visualization from the last epoch
                        if epoch == args.max_epoch - 1:
                            if args.confusion_matrix or args.t_SNE:

                                if phase == 'source_val':
                                    if tsne_i_source == 0:
                                        if args.which_classifier == 'two_layers_tsne':
                                            features_tsne_source_G1 = features_two_layers_tsne_G1
                                            features_tsne_source_G2 = features_two_layers_tsne_G2


                                        label_tsne_source = labels
                                        tsne_i_source = 1

                                    else:
                                        if args.which_classifier == 'two_layers_tsne':
                                            features_tsne_source_G1 = torch.cat((features_tsne_source_G1, features_two_layers_tsne_G1))
                                            features_tsne_source_G2 = torch.cat((features_tsne_source_G2, features_two_layers_tsne_G2))


                                        label_tsne_source = torch.cat((label_tsne_source, labels))

                                if phase == 'target_val':
                                    if tsne_i_target == 0:
                                        if args.which_classifier == 'two_layers_tsne':
                                            features_tsne_target_G1 = features_two_layers_tsne_G1
                                            features_tsne_target_G2 = features_two_layers_tsne_G2


                                        label_tsne_target = labels
                                        tsne_i_target = 1

                                    else:
                                        if args.which_classifier == 'two_layers_tsne':
                                            features_tsne_target_G1 = torch.cat((features_tsne_target_G1, features_two_layers_tsne_G1))
                                            features_tsne_target_G2 = torch.cat((features_tsne_target_G2, features_two_layers_tsne_G2))


                                        label_tsne_target = torch.cat((label_tsne_target, labels))

                                    pred_label_F1, pred_label_F2, pred_label_Avg = pred_F1.cpu().numpy(), pred_F2.cpu().numpy(), pred_avg.cpu().numpy()
                                    true_label = labels.cpu().numpy()

                                    for cm in range(labels.size(0)):
                                        pred_label_cm = pred_label_F1[cm]
                                        true_label_cm = true_label[cm]
                                        confusion_matrix_F1[pred_label_cm][true_label_cm] += 1

                                    for cm in range(labels.size(0)):
                                        pred_label_cm = pred_label_F2[cm]
                                        true_label_cm = true_label[cm]
                                        confusion_matrix_F2[pred_label_cm][true_label_cm] += 1

                                    for cm in range(labels.size(0)):
                                        pred_label_cm = pred_label_Avg[cm]
                                        true_label_cm = true_label[cm]
                                        confusion_matrix_Avg[pred_label_cm][true_label_cm] += 1



                epoch_loss = epoch_loss / epoch_length
                epoch_acc = epoch_acc / epoch_length

                # acc%
                epoch_acc = epoch_acc * 100


                if phase == 'source_train':
                    source_train_loss_pic.append(epoch_loss)
                    source_train_acc_pic.append((epoch_acc))
                elif phase == 'source_val':
                    source_val_loss_pic.append(epoch_loss)
                    source_val_acc_pic.append((epoch_acc))
                elif phase == 'target_val':
                    target_val_loss_pic.append(epoch_loss)
                    target_val_acc_pic.append((epoch_acc))

                    epoch_acc_F1 = epoch_acc_F1 / epoch_length
                    epoch_acc_F2 = epoch_acc_F2 / epoch_length
                    epoch_acc_Avg = epoch_acc_Avg / epoch_length

                    #
                    epoch_acc_F1 = epoch_acc_F1 * 100
                    epoch_acc_F2 = epoch_acc_F2 * 100
                    epoch_acc_Avg = epoch_acc_Avg * 100





                #
                if phase == 'target_val':

                    if epoch == args.max_epoch - 1:
                        last_acc_F1 = epoch_acc_F1
                        last_acc_F2 = epoch_acc_F2
                        last_acc_Avg = epoch_acc_Avg


            gc.collect()
            torch.cuda.empty_cache()


            if self.lr_scheduler_F1 is not None:
                self.lr_scheduler_F1.step()
            if self.lr_scheduler_G1 is not None:
                self.lr_scheduler_G1.step()
            if self.lr_scheduler_F2 is not None:
                self.lr_scheduler_F2.step()
            if self.lr_scheduler_G2 is not None:
                self.lr_scheduler_G2.step()


        print(Dataset.num_classes, "category task", end='     ')
        print("training time：", time.time() - start_time, end='     ')
        print(Fore.RED + "F1 final acc", last_acc_F1, end='     ')
        print(Fore.RED + "F2 final acc", last_acc_F2, end='     ')
        print(Fore.RED + "Avg final acc", last_acc_Avg, end='     ')
        print(Style.RESET_ALL)



        plt.figure(1)
        plt.ion()
        plt.subplot(211)
        plt.plot(x_for_pic, source_train_loss_pic, label="source_train_loss")
        plt.plot(x_for_pic, source_val_loss_pic, label="source_val_loss")
        plt.plot(x_for_pic, target_val_loss_pic, label="target_val_loss")
        plt.legend()
        plt.title('Loss')
        plt.figure(1)
        plt.subplot(212)
        plt.plot(x_for_pic, source_train_acc_pic, label="source_train_acc")
        plt.plot(x_for_pic, source_val_acc_pic, label="source_val_acc")
        plt.plot(x_for_pic, target_val_acc_pic, label="target_val_acc")
        plt.legend()
        plt.title('Acc')
        plt.xlabel('epoch')

        pic_form_dir = args.pic_form_dir
        pic_sub_dir1 = auto_transfer_task
        pic_sub_dir2 = time_repeat
        pic_sub_dir1 = str(pic_sub_dir1)
        pic_sub_dir2 = str(pic_sub_dir2)
        pic_sub_dir = 'P' + pic_sub_dir1 + '_' + pic_sub_dir2
        pic_dir = pic_form_dir + '\\' + pic_sub_dir
        plt.ioff()
        plt.savefig(pic_dir)
        plt.clf()

        if args.save_array:
            # source_train_loss
            array_dir_stl = pic_form_dir + '\\' + 'Array' + pic_sub_dir1 + '_' + pic_sub_dir2 + 'stl.csv'
            source_train_loss_pic = np.array(source_train_loss_pic)
            np.savetxt(array_dir_stl, source_train_loss_pic, delimiter=',')

            # source_val_loss
            array_dir_svl = pic_form_dir + '\\' + 'Array' + pic_sub_dir1 + '_' + pic_sub_dir2 + 'svl.csv'
            source_val_loss_pic = np.array(source_val_loss_pic)
            np.savetxt(array_dir_svl, source_val_loss_pic, delimiter=',')

            # target_val_loss
            array_dir_tvl = pic_form_dir + '\\' + 'Array' + pic_sub_dir1 + '_' + pic_sub_dir2 + 'tvl.csv'
            target_val_loss_pic = np.array(target_val_loss_pic)
            np.savetxt(array_dir_tvl, target_val_loss_pic, delimiter=',')

            # source_train_acc
            array_dir_sta = pic_form_dir + '\\' + 'Array' + pic_sub_dir1 + '_' + pic_sub_dir2 + 'sta.csv'
            source_train_acc_pic = np.array(source_train_acc_pic)
            np.savetxt(array_dir_sta, source_train_acc_pic, delimiter=',')

            # source_val_acc
            array_dir_sva = pic_form_dir + '\\' + 'Array' + pic_sub_dir1 + '_' + pic_sub_dir2 + 'sva.csv'
            source_val_acc_pic = np.array(source_val_acc_pic)
            np.savetxt(array_dir_sva, source_val_acc_pic, delimiter=',')

            # target_val_acc
            array_dir_tva = pic_form_dir + '\\' + 'Array' + pic_sub_dir1 + '_' + pic_sub_dir2 + 'tva.csv'
            target_val_acc_pic = np.array(target_val_acc_pic)
            np.savetxt(array_dir_tva, target_val_acc_pic, delimiter=',')


        # cm  F1
        if args.confusion_matrix:

            plt.figure(3)
            plt.ion()
            num_class_cm = Dataset.num_classes
            if "CWRU" in args.data_name:
                confusion_matrix_class =['H','IR7','B7','OR7','IR14','B14','OR14','IR21','B21','OR21']
            if "JNU" in args.data_name:
                confusion_matrix_class =['IB','N','OB','TB']
            if "PHM" in args.data_name:
                confusion_matrix_class =['H1','H2','H3','H4','H5','H6']
            if "PU" in args.data_name: #RDB
                confusion_matrix_class =['KA04','KA15','KA16','KA22','KA30','KB23','KB24','KB27','KI14','KI16','KI17','KI18','KI21']
            if "SEU" in args.data_name:
                confusion_matrix_class =['H','BA','OU','IN','CO','CI','MI','SU','RO']
            if "PT500" in args.data_name:
                confusion_matrix_class = ['N', 'B', 'C', 'I', 'O']
            if "PT1" in args.data_name:
                confusion_matrix_class = ['N', 'B', 'C', 'I', 'O']
            if "PT2" in args.data_name:
                confusion_matrix_class = ['N', 'B', 'C', 'I', 'O']
            if "HUST_bearings" in args.data_name:
                confusion_matrix_class = ['B', 'I', 'O', 'H', 'C']
            if "BJTU_RAO_B" in args.data_name:
                confusion_matrix_class = ['G0', 'G1', 'G4', 'G5', 'G7']

            if args.conf_matrix_normalize:
                confusion_matrix = confusion_matrix_F1.astype('float') / confusion_matrix_F1.sum(axis=0)[:, np.newaxis]
                confusion_matrix = confusion_matrix * 100
                confusion_matrix = confusion_matrix
                fmt = '.2f'
            else:
                confusion_matrix = np.asarray(confusion_matrix_F1, dtype=int)
                fmt = 'd'

            # confusion_matrix = np.nan_to_num(confusion_matrix, nan=0.00)

            plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Purples)  # Blues   Greys    Purples
            plt.title('confusion_matrix')
            plt.colorbar()
            tick_marks = np.arange(num_class_cm)
            plt.xticks(tick_marks, confusion_matrix_class, rotation=0)
            plt.yticks(tick_marks, confusion_matrix_class)
            thresh = confusion_matrix.max() / 2.
            for i, j in itertools.product(range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])):
                plt.text(j, i, format(confusion_matrix[i, j], fmt),
                         horizontalalignment="center",
                         color="white" if confusion_matrix[i, j] > thresh else "black",
                         fontsize = 17)
            plt.tight_layout()
            plt.xlabel('True label')
            plt.ylabel('Predicted label')
            # plt.title('{}{}{}{}{}{}'.format("Confu matrix    Data-", args.data_name, "    Task-", transfer_task, "    Last-Acc",epoch_acc))
            pic_form_dir = args.pic_form_dir
            pic_sub_dir1 = auto_transfer_task
            pic_sub_dir2 = time_repeat
            pic_sub_dir1 = str(pic_sub_dir1)
            pic_sub_dir2 = str(pic_sub_dir2)
            pic_sub_dir = 'CM_F1' + pic_sub_dir1 + '_' + pic_sub_dir2
            pic_dir = pic_form_dir + '\\' + pic_sub_dir + '.svg'
            plt.ioff()
            plt.savefig(pic_dir, format='svg')
            plt.clf()



        # cm  F2
        if args.confusion_matrix:

            plt.figure(4)
            plt.ion()
            num_class_cm = Dataset.num_classes
            if "CWRU" in args.data_name:
                confusion_matrix_class = ['H', 'IR7', 'B7', 'OR7', 'IR14', 'B14', 'OR14', 'IR21', 'B21', 'OR21']
            if "JNU" in args.data_name:
                confusion_matrix_class = ['IB', 'N', 'OB', 'TB']
            if "PHM" in args.data_name:
                confusion_matrix_class = ['H1', 'H2', 'H3', 'H4', 'H5', 'H6']
            if "PU" in args.data_name:  # RDB
                confusion_matrix_class = ['KA04', 'KA15', 'KA16', 'KA22', 'KA30', 'KB23', 'KB24', 'KB27', 'KI14',
                                          'KI16', 'KI17', 'KI18', 'KI21']
            if "SEU" in args.data_name:
                confusion_matrix_class = ['H', 'BA', 'OU', 'IN', 'CO', 'CI', 'MI', 'SU', 'RO']
            if "PT500" in args.data_name:
                confusion_matrix_class = ['N', 'B', 'C', 'I', 'O']
            if "PT1" in args.data_name:
                confusion_matrix_class = ['N', 'B', 'C', 'I', 'O']
            if "PT2" in args.data_name:
                confusion_matrix_class = ['N', 'B', 'C', 'I', 'O']
            if "HUST_bearings" in args.data_name:
                confusion_matrix_class = ['B', 'I', 'O', 'H', 'C']
            if "BJTU_RAO_B" in args.data_name:
                confusion_matrix_class = ['G0', 'G1', 'G4', 'G5', 'G7']

            if args.conf_matrix_normalize:
                confusion_matrix = confusion_matrix_F2.astype('float') / confusion_matrix_F2.sum(axis=0)[:, np.newaxis]
                confusion_matrix = confusion_matrix * 100
                confusion_matrix = confusion_matrix
                fmt = '.2f'
            else:
                confusion_matrix = np.asarray(confusion_matrix_F2, dtype=int)
                fmt = 'd'

            # confusion_matrix = np.nan_to_num(confusion_matrix, nan=0.00)

            plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Purples)  # Blues   Greys    Purples
            plt.title('confusion_matrix')
            plt.colorbar()
            tick_marks = np.arange(num_class_cm)
            plt.xticks(tick_marks, confusion_matrix_class, rotation=0)
            plt.yticks(tick_marks, confusion_matrix_class)
            thresh = confusion_matrix.max() / 2.
            for i, j in itertools.product(range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])):
                plt.text(j, i, format(confusion_matrix[i, j], fmt),
                         horizontalalignment="center",
                         color="white" if confusion_matrix[i, j] > thresh else "black",
                         fontsize=17)
            plt.tight_layout()
            plt.xlabel('True label')
            plt.ylabel('Predicted label')
            # plt.title('{}{}{}{}{}{}'.format("Confu matrix    Data-", args.data_name, "    Task-", transfer_task, "    Last-Acc",epoch_acc))
            pic_form_dir = args.pic_form_dir
            pic_sub_dir1 = auto_transfer_task
            pic_sub_dir2 = time_repeat
            pic_sub_dir1 = str(pic_sub_dir1)
            pic_sub_dir2 = str(pic_sub_dir2)
            pic_sub_dir = 'CM_F2' + pic_sub_dir1 + '_' + pic_sub_dir2
            pic_dir = pic_form_dir + '\\' + pic_sub_dir + '.svg'
            plt.ioff()
            plt.savefig(pic_dir, format='svg')
            plt.clf()

        # cm  Avg
        if args.confusion_matrix:

            plt.figure(5)
            plt.ion()
            num_class_cm = Dataset.num_classes
            if "CWRU" in args.data_name:
                confusion_matrix_class = ['H', 'IR7', 'B7', 'OR7', 'IR14', 'B14', 'OR14', 'IR21', 'B21', 'OR21']
            if "JNU" in args.data_name:
                confusion_matrix_class = ['IB', 'N', 'OB', 'TB']
            if "PHM" in args.data_name:
                confusion_matrix_class = ['H1', 'H2', 'H3', 'H4', 'H5', 'H6']
            if "PU" in args.data_name:  # RDB
                confusion_matrix_class = ['KA04', 'KA15', 'KA16', 'KA22', 'KA30', 'KB23', 'KB24', 'KB27', 'KI14',
                                          'KI16', 'KI17', 'KI18', 'KI21']
            if "SEU" in args.data_name:
                confusion_matrix_class = ['H', 'BA', 'OU', 'IN', 'CO', 'CI', 'MI', 'SU', 'RO']
            if "PT500" in args.data_name:
                confusion_matrix_class = ['N', 'B', 'C', 'I', 'O']
            if "PT1" in args.data_name:
                confusion_matrix_class = ['N', 'B', 'C', 'I', 'O']
            if "PT2" in args.data_name:
                confusion_matrix_class = ['N', 'B', 'C', 'I', 'O']
            if "HUST_bearings" in args.data_name:
                confusion_matrix_class = ['B', 'I', 'O', 'H', 'C']
            if "BJTU_RAO_B" in args.data_name:
                confusion_matrix_class = ['G0', 'G1', 'G4', 'G5', 'G7']

            if args.conf_matrix_normalize:
                confusion_matrix = confusion_matrix_Avg.astype('float') / confusion_matrix_Avg.sum(axis=0)[:, np.newaxis]
                confusion_matrix = confusion_matrix * 100
                confusion_matrix = confusion_matrix
                fmt = '.2f'
            else:
                confusion_matrix = np.asarray(confusion_matrix_Avg, dtype=int)
                fmt = 'd'

            # confusion_matrix = np.nan_to_num(confusion_matrix, nan=0.00)

            plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Purples)  # Blues   Greys    Purples
            plt.title('confusion_matrix')
            plt.colorbar()
            tick_marks = np.arange(num_class_cm)
            plt.xticks(tick_marks, confusion_matrix_class, rotation=0)
            plt.yticks(tick_marks, confusion_matrix_class)
            thresh = confusion_matrix.max() / 2.
            for i, j in itertools.product(range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])):
                plt.text(j, i, format(confusion_matrix[i, j], fmt),
                         horizontalalignment="center",
                         color="white" if confusion_matrix[i, j] > thresh else "black",
                         fontsize=17)
            plt.tight_layout()
            plt.xlabel('True label')
            plt.ylabel('Predicted label')
            # plt.title('{}{}{}{}{}{}'.format("Confu matrix    Data-", args.data_name, "    Task-", transfer_task, "    Last-Acc",epoch_acc))
            pic_form_dir = args.pic_form_dir
            pic_sub_dir1 = auto_transfer_task
            pic_sub_dir2 = time_repeat
            pic_sub_dir1 = str(pic_sub_dir1)
            pic_sub_dir2 = str(pic_sub_dir2)
            pic_sub_dir = 'CM_Avg' + pic_sub_dir1 + '_' + pic_sub_dir2
            pic_dir = pic_form_dir + '\\' + pic_sub_dir + '.svg'
            plt.ioff()
            plt.savefig(pic_dir, format='svg')
            plt.clf()



        # t-SNE
        if args.t_SNE:
            #
            features_tsne_G1 = torch.cat((features_tsne_source_G1, features_tsne_target_G1))
            features_tsne_G1 = features_tsne_G1.cpu().numpy()

            features_tsne_G2 = torch.cat((features_tsne_source_G2, features_tsne_target_G2))
            features_tsne_G2 = features_tsne_G2.cpu().numpy()

            #
            labels_tsne = torch.cat((label_tsne_source,label_tsne_target))
            labels_tsne = labels_tsne.cpu().numpy()
            time_t_SNE_start = time.time()

            #
            if "JNU" in args.data_name:
                legend_dataset =['I F','NA','OF','BF']
            if "PT1TL" in args.data_name:
                legend_dataset = ['N', 'I', 'O', 'B', 'C']
            if "PT2TL" in args.data_name:
                legend_dataset = ['N', 'I', 'O', 'B', 'C']
            if "HUST_bearings" in args.data_name:
                legend_dataset = ['N', 'B', 'C', 'I', 'O']
            if "BJTU_RAO_B" in args.data_name:
                legend_dataset = ['G0', 'G1', 'G4', 'G5', 'G7']

            #　G1
            if args.t_SNE_2D:

                #print("Computing 2D-t-SNE embedding")
                #matplotlib.use('Agg')
                tsne2d = TSNE(n_components=2, init='pca', random_state=0)
                X_tsne_2d = tsne2d.fit_transform(features_tsne_G1)
                x_min, x_max = np.min(X_tsne_2d[:, 0:2], 0), np.max(X_tsne_2d[:, 0:2], 0)
                X = (X_tsne_2d[:, 0:2] - x_min) / (x_max - x_min)
                plt.figure()    #figsize=(20, 15)

                # autocolor mode
                if args.t_SNE_auto_color:
                    #
                    for j in range(Dataset.num_classes):
                        s = 0
                        #
                        for i in range(source_val_length):
                            if (j == labels_tsne[i]) & (s == 0):
                                label = 'S' + '-' + str(legend_dataset[labels_tsne[i]])
                                plt.scatter(X[i, 0], X[i, 1], marker='+', color=plt.cm.Set3(labels_tsne[i] / Dataset.num_classes),
                                            label=label)
                                s = 1
                            if j == labels_tsne[i]:
                                plt.scatter(X[i, 0], X[i, 1], marker='+', color=plt.cm.Set3(labels_tsne[i] / Dataset.num_classes))

                    for j in range(Dataset.num_classes):
                        s = 0
                        for i in range(source_val_length, X.shape[0]):
                            if (j == labels_tsne[i]) & (s == 0):
                                label = 'T' + '-' + str(legend_dataset[labels_tsne[i]])
                                plt.scatter(X[i, 0], X[i, 1], marker='o', color=plt.cm.Set3(labels_tsne[i] / Dataset.num_classes),
                                            label=label)
                                s = 1
                            if j == labels_tsne[i]:
                                plt.scatter(X[i, 0], X[i, 1], marker='o', color=plt.cm.Set3(labels_tsne[i] / Dataset.num_classes))

                # manual color mode
                else:

                    # 颜色：color=plt.cm.Set3(labels_tsne[i] / Dataset.num_classes * color_bar)
                    # 0浅绿色    0.1黄色   0.2魔芋紫  0.3火烈鸟红  0.4蓝色
                    # 0.5荧光绿  0.6粉色   0.7灰色    0.8紫色     0.9浅绿色

                    color_bar = [0.3, 0.4, 0.5, 0.6, 0.8,]

                    #
                    for j in range(Dataset.num_classes):
                        s = 0
                        #
                        for i in range(source_val_length):
                            if (j == labels_tsne[i]) & (s == 0):
                                label = 'S' + '-' + str(legend_dataset[labels_tsne[i]])
                                plt.scatter(X[i, 0], X[i, 1], marker='+', color=plt.cm.Set3(color_bar[labels_tsne[i]]),
                                            label=label)
                                s = 1
                            if j == labels_tsne[i]:
                                plt.scatter(X[i, 0], X[i, 1], marker='+', color=plt.cm.Set3(color_bar[labels_tsne[i]]))

                    for j in range(Dataset.num_classes):
                        s = 0
                        for i in range(source_val_length, X.shape[0]):
                            if (j == labels_tsne[i]) & (s == 0):
                                label = 'T' + '-' + str(legend_dataset[labels_tsne[i]])
                                plt.scatter(X[i, 0], X[i, 1], marker='o', color=plt.cm.Set3(color_bar[labels_tsne[i]]),
                                            label=label)
                                s = 1
                            if j == labels_tsne[i]:
                                plt.scatter(X[i, 0], X[i, 1], marker='o', color=plt.cm.Set3(color_bar[labels_tsne[i]]))

                # plt.legend(loc='best', frameon=False, ncol=12)
                # plt.title('{}{}{}{}{}{}'.format("2D-tSNE    Data-", args.data_name, "    Task-", transfer_task, "    Last-Acc", epoch_acc))
                pic_form_dir = args.pic_form_dir
                pic_sub_dir1 = auto_transfer_task
                pic_sub_dir2 = time_repeat
                pic_sub_dir1 = str(pic_sub_dir1)
                pic_sub_dir2 = str(pic_sub_dir2)
                pic_sub_dir = 'ECB_G1_T2' + pic_sub_dir1 + '_' + pic_sub_dir2
                pic_dir = pic_form_dir + '\\' + pic_sub_dir + '.svg'
                plt.savefig(pic_dir, format='svg')
                plt.clf()


            if args.t_SNE_3D:

                #print("Computing 3D-t-SNE embedding")
                #matplotlib.use('Agg')
                tsne3d = TSNE(n_components=3, init='pca', random_state=0)
                X_tsne_3d = tsne3d.fit_transform(features_tsne_G1)
                x_min, x_max = np.min(X_tsne_3d[:,0:3], axis=0), np.max(X_tsne_3d[:,0:3], axis=0)
                X = (X_tsne_3d[:,0:3] - x_min) / (x_max - x_min)
                fig = plt.figure()
                ax = Axes3D(fig)
                #
                for j in range(Dataset.num_classes):
                    s = 0
                    #
                    for i in range(source_val_length):
                        if (j == labels_tsne[i]) & (s == 0):
                            label = 'S' + '-' + str(legend_dataset[labels_tsne[i]])
                            ax.plot3D(X[i, 0], X[i, 1], X[i, 2], marker='+',color=plt.cm.Set3(labels_tsne[i] / Dataset.num_classes),
                                        label = label)
                            s = 1
                        if j == labels_tsne[i]:
                            ax.plot3D(X[i, 0], X[i, 1], X[i, 2], marker='+',color=plt.cm.Set3(labels_tsne[i] / Dataset.num_classes))
                for j in range(Dataset.num_classes):
                    s = 0
                    for i in range(source_val_length, X.shape[0]):
                        if (j == labels_tsne[i]) & (s == 0):
                            label = 'T' + '-' + str(legend_dataset[labels_tsne[i]])
                            ax.plot3D(X[i, 0], X[i, 1], X[i, 2], marker='o',color=plt.cm.Set3(labels_tsne[i] / Dataset.num_classes),
                                        label = label)
                            s = 1
                        if j == labels_tsne[i]:
                            ax.plot3D(X[i, 0], X[i, 1], X[i, 2], marker='o',color=plt.cm.Set3(labels_tsne[i] / Dataset.num_classes))
                ax.legend(loc='best', frameon=False, ncol=2)
                # ax.text3D(0, 0, 1.3, '{}{}{}{}{}{}'.format("3D-tSNE    Data-", args.data_name, "    Task-", transfer_task, "    Last-Acc", epoch_acc))

                pic_form_dir = args.pic_form_dir
                pic_sub_dir1 = auto_transfer_task
                pic_sub_dir2 = time_repeat
                pic_sub_dir1 = str(pic_sub_dir1)
                pic_sub_dir2 = str(pic_sub_dir2)
                pic_sub_dir = 'ECB_G1_T3' + pic_sub_dir1 + '_' + pic_sub_dir2
                pic_dir = pic_form_dir + '\\' + pic_sub_dir + '.svg'
                plt.savefig(pic_dir)
                plt.clf()



            # 　G2
            if args.t_SNE_2D:

                # print("Computing 2D-t-SNE embedding")
                # matplotlib.use('Agg')
                tsne2d = TSNE(n_components=2, init='pca', random_state=0)
                X_tsne_2d = tsne2d.fit_transform(features_tsne_G2)
                x_min, x_max = np.min(X_tsne_2d[:, 0:2], 0), np.max(X_tsne_2d[:, 0:2], 0)
                X = (X_tsne_2d[:, 0:2] - x_min) / (x_max - x_min)
                plt.figure()  # figsize=(20, 15)

                # auto color
                if args.t_SNE_auto_color:
                    # 每个类别
                    for j in range(Dataset.num_classes):
                        s = 0
                        #  每个样本
                        for i in range(source_val_length):
                            if (j == labels_tsne[i]) & (s == 0):
                                label = 'S' + '-' + str(legend_dataset[labels_tsne[i]])
                                plt.scatter(X[i, 0], X[i, 1], marker='+',
                                            color=plt.cm.Set3(labels_tsne[i] / Dataset.num_classes),
                                            label=label)
                                s = 1
                            if j == labels_tsne[i]:
                                plt.scatter(X[i, 0], X[i, 1], marker='+',
                                            color=plt.cm.Set3(labels_tsne[i] / Dataset.num_classes))

                    for j in range(Dataset.num_classes):
                        s = 0
                        for i in range(source_val_length, X.shape[0]):
                            if (j == labels_tsne[i]) & (s == 0):
                                label = 'T' + '-' + str(legend_dataset[labels_tsne[i]])
                                plt.scatter(X[i, 0], X[i, 1], marker='o',
                                            color=plt.cm.Set3(labels_tsne[i] / Dataset.num_classes),
                                            label=label)
                                s = 1
                            if j == labels_tsne[i]:
                                plt.scatter(X[i, 0], X[i, 1], marker='o',
                                            color=plt.cm.Set3(labels_tsne[i] / Dataset.num_classes))

                # manual color
                else:

                    # 颜色：color=plt.cm.Set3(labels_tsne[i] / Dataset.num_classes * color_bar)
                    # 0浅绿色    0.1黄色   0.2魔芋紫  0.3火烈鸟红  0.4蓝色
                    # 0.5荧光绿  0.6粉色   0.7灰色    0.8紫色     0.9浅绿色

                    color_bar = [0.3, 0.4, 0.5, 0.6, 0.8, ]

                    # 每个类别
                    for j in range(Dataset.num_classes):
                        s = 0
                        #  每个样本
                        for i in range(source_val_length):
                            if (j == labels_tsne[i]) & (s == 0):
                                label = 'S' + '-' + str(legend_dataset[labels_tsne[i]])
                                plt.scatter(X[i, 0], X[i, 1], marker='+',
                                            color=plt.cm.Set3(color_bar[labels_tsne[i]]),
                                            label=label)
                                s = 1
                            if j == labels_tsne[i]:
                                plt.scatter(X[i, 0], X[i, 1], marker='+',
                                            color=plt.cm.Set3(color_bar[labels_tsne[i]]))

                    for j in range(Dataset.num_classes):
                        s = 0
                        for i in range(source_val_length, X.shape[0]):
                            if (j == labels_tsne[i]) & (s == 0):
                                label = 'T' + '-' + str(legend_dataset[labels_tsne[i]])
                                plt.scatter(X[i, 0], X[i, 1], marker='o',
                                            color=plt.cm.Set3(color_bar[labels_tsne[i]]),
                                            label=label)
                                s = 1
                            if j == labels_tsne[i]:
                                plt.scatter(X[i, 0], X[i, 1], marker='o',
                                            color=plt.cm.Set3(color_bar[labels_tsne[i]]))

                # plt.legend(loc='best', frameon=False, ncol=12)
                # plt.title('{}{}{}{}{}{}'.format("2D-tSNE    Data-", args.data_name, "    Task-", transfer_task, "    Last-Acc", epoch_acc))
                pic_form_dir = args.pic_form_dir
                pic_sub_dir1 = auto_transfer_task
                pic_sub_dir2 = time_repeat
                pic_sub_dir1 = str(pic_sub_dir1)
                pic_sub_dir2 = str(pic_sub_dir2)
                pic_sub_dir = 'ECB_G2_T2' + pic_sub_dir1 + '_' + pic_sub_dir2
                pic_dir = pic_form_dir + '\\' + pic_sub_dir + '.svg'
                plt.savefig(pic_dir, format='svg')
                plt.clf()


            if args.t_SNE_3D:

                # print("Computing 3D-t-SNE embedding")
                # matplotlib.use('Agg')
                tsne3d = TSNE(n_components=3, init='pca', random_state=0)
                X_tsne_3d = tsne3d.fit_transform(features_tsne_G2)
                x_min, x_max = np.min(X_tsne_3d[:, 0:3], axis=0), np.max(X_tsne_3d[:, 0:3], axis=0)
                X = (X_tsne_3d[:, 0:3] - x_min) / (x_max - x_min)
                fig = plt.figure()
                ax = Axes3D(fig)
                #
                for j in range(Dataset.num_classes):
                    s = 0
                    #
                    for i in range(source_val_length):
                        if (j == labels_tsne[i]) & (s == 0):
                            label = 'S' + '-' + str(legend_dataset[labels_tsne[i]])
                            ax.plot3D(X[i, 0], X[i, 1], X[i, 2], marker='+',
                                      color=plt.cm.Set3(labels_tsne[i] / Dataset.num_classes),
                                      label=label)
                            s = 1
                        if j == labels_tsne[i]:
                            ax.plot3D(X[i, 0], X[i, 1], X[i, 2], marker='+',
                                      color=plt.cm.Set3(labels_tsne[i] / Dataset.num_classes))
                for j in range(Dataset.num_classes):
                    s = 0
                    for i in range(source_val_length, X.shape[0]):
                        if (j == labels_tsne[i]) & (s == 0):
                            label = 'T' + '-' + str(legend_dataset[labels_tsne[i]])
                            ax.plot3D(X[i, 0], X[i, 1], X[i, 2], marker='o',
                                      color=plt.cm.Set3(labels_tsne[i] / Dataset.num_classes),
                                      label=label)
                            s = 1
                        if j == labels_tsne[i]:
                            ax.plot3D(X[i, 0], X[i, 1], X[i, 2], marker='o',
                                      color=plt.cm.Set3(labels_tsne[i] / Dataset.num_classes))
                ax.legend(loc='best', frameon=False, ncol=2)
                # ax.text3D(0, 0, 1.3, '{}{}{}{}{}{}'.format("3D-tSNE    Data-", args.data_name, "    Task-", transfer_task, "    Last-Acc", epoch_acc))
                pic_form_dir = args.pic_form_dir
                pic_sub_dir1 = auto_transfer_task
                pic_sub_dir2 = time_repeat
                pic_sub_dir1 = str(pic_sub_dir1)
                pic_sub_dir2 = str(pic_sub_dir2)
                pic_sub_dir = 'ECB_G2_T3' + pic_sub_dir1 + '_' + pic_sub_dir2
                pic_dir = pic_form_dir + '\\' + pic_sub_dir + '.svg'
                plt.savefig(pic_dir, format='svg')
                plt.clf()




        #plt.ioff()
        plt.show()
        #


        return last_acc_F1, last_acc_F2, last_acc_Avg