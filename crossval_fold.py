import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F

import numpy as np
import sys
import os

from sklearn.utils import class_weight

from data_loader import data_loader
from data_loader import CustomDataset
from model.tcn import TemporalConvNet
from model import metric
from utils import early_stopping
import main_TCN_Liu
from main_TCN_Liu import TCN

import wandb


def cross_val_train(num_of_fold, X_train_fold, y_train_fold, X_valid_fold,
                    y_valid_fold, X_test_fold, y_test_fold, args, nclass,
                    device):
    max_recall0 = []
    max_precision0 = []
    max_recall1 = []
    max_precision1 = []
    max_acc = []
    max_tss = []
    max_bacc = []
    max_hss = []
    train_recall1list = []
    train_precision1list = []
    train_acclist = []
    train_hsslist = []
    train_tsslist = []
    train_bacclist = []
    val_recall1list = []
    val_precision1list = []
    val_acclist = []
    val_hsslist = []
    val_tsslist = []
    val_bacclist = []
    test_recall1list = []
    test_precision1list = []
    test_acclist = []
    test_hsslist = []
    test_tsslist = []
    test_bacclist = []

    for train_itr in range(num_of_fold):
        X_train = []
        y_train = []
        for j in range(num_of_fold):
            if j != train_itr:
                for k in range(len(X_train_fold[j])):
                    X_train.append(X_train_fold[j][k])
                    y_train.append(y_train_fold[j][k])

        for test_itr in range(num_of_fold):
            print('------------- ' + str(
                train_itr * num_of_fold + test_itr) + ' iteration----------------')
            X_valid = []
            y_valid = []
            X_test = []
            y_test = []
            for j in range(num_of_fold):
                if j != test_itr:
                    for k in range(len(X_valid_fold[j])):
                        X_valid.append(X_valid_fold[j][k])
                        y_valid.append(y_valid_fold[j][k])
                    for k in range(len(X_test_fold[j])):
                        X_test.append(X_test_fold[j][k])
                        y_test.append(y_test_fold[j][k])

            X_train = np.array(X_train)
            y_train = np.array(y_train)
            X_valid = np.array(X_valid)
            y_valid = np.array(y_valid)
            X_test = np.array(X_test)
            y_test = np.array(y_test)

            y_train_tr = data_loader.label_transform(y_train)
            y_valid_tr = data_loader.label_transform(y_valid)
            y_test_tr = data_loader.label_transform(y_test)

            # (samples, seq_len, features) -> (samples, features, seq_len)
            X_train_data_tensor = torch.tensor(X_train).float()
            X_train_data_tensor = X_train_data_tensor.permute(0, 2, 1)
            y_train_tr_tensor = torch.tensor(y_train_tr).long()

            X_valid_data_tensor = torch.tensor(X_valid).float()
            X_valid_data_tensor = X_valid_data_tensor.permute(0, 2, 1)
            y_valid_tr_tensor = torch.tensor(y_valid_tr).long()

            X_test_data_tensor = torch.tensor(X_test).float()
            X_test_data_tensor = X_test_data_tensor.permute(0, 2, 1)
            y_test_tr_tensor = torch.tensor(y_test_tr).long()

            # ready custom dataset
            datasets = {'train': main_TCN_Liu.preprocess_customdataset(
                X_train_data_tensor, y_train_tr_tensor),
                'valid': main_TCN_Liu.preprocess_customdataset(
                    X_valid_data_tensor, y_valid_tr_tensor),
                'test': main_TCN_Liu.preprocess_customdataset(
                    X_test_data_tensor, y_test_tr_tensor)}

            kwargs = {'num_workers': args.num_workers,
                      'pin_memory': True} if args.cuda else {}

            train_loader = torch.utils.data.DataLoader(datasets['train'],
                                                       args.batch_size,
                                                       shuffle=False,
                                                       drop_last=False,
                                                       **kwargs)
            valid_loader = torch.utils.data.DataLoader(datasets['valid'],
                                                       args.batch_size,
                                                       shuffle=False,
                                                       drop_last=False,
                                                       **kwargs)
            test_loader = torch.utils.data.DataLoader(datasets['test'],
                                                      args.batch_size,
                                                      shuffle=False,
                                                      drop_last=False,
                                                      **kwargs)
            # Shape: (batch size, features, seq_len)
            # make model
            channel_sizes = [args.nhid] * args.levels
            kernel_size = args.ksize
            dropout = args.dropout

            model = TCN(args.n_features, nclass, channel_sizes,
                        kernel_size=kernel_size, dropout=dropout).to(device)
            wandb.watch(model, log='all')

            # optimizers
            class_weights = class_weight.compute_class_weight('balanced',
                                                              np.unique(
                                                                  y_train),
                                                              y_train)

            # noinspection PyArgumentList
            criterion = nn.CrossEntropyLoss(
                weight=torch.FloatTensor(class_weights).to(
                    device))  # weighted cross entropy
            optimizer = torch.optim.Adam(model.parameters(),
                                         lr=args.learning_rate,
                                         weight_decay=args.weight_decay,
                                         amsgrad=False)

            # print model parameters
            print("Receptive Field: " + str(
                1 + 2 * (args.ksize - 1) * (2 ** args.levels - 1)))

            # early stopping check
            early_stop = early_stopping.EarlyStopping(mode='max', patience=30)
            best_tss = 0.0
            best_pr_auc = 0.0
            best_epoch = 0
            epoch = 0

            val_recall, val_precision, val_accuracy,\
            val_bacc, val_hss, val_tss = 0, 0, 0, 0, 0, 0


            print('{:<11s}{:^9s}{:^9s}{:^9s}'
                  '{:^9s}{:^9s}{:^9s}{:^9s}'
                  '{:^9s}{:^9s}'
                  '{:^9s}{:^9s}{:^3s}'.format('Data Loader', 'Epoch',
                                              'Runtime', 'TSS', 'PR_AUC',
                                              'HSS', 'BACC', 'ACC',
                                              'Precision', 'Recall', 'F1',
                                              'Loss', 'CP'))

            if args.training:
                while epoch < args.epochs:
                    train_recall, train_precision, train_accuracy, \
                    train_bacc, train_hss, train_tss = main_TCN_Liu.train(
                        model, device, train_loader, optimizer, epoch,
                        criterion, args)

                    stopping_metric, best_tss, best_pr_auc, best_epoch, \
                    val_recall, val_precision, val_accuracy, val_bacc, \
                    val_hss, val_tss = main_TCN_Liu.validate(model, device,
                                                             valid_loader,
                                                             criterion, epoch,
                                                             best_tss,
                                                             best_pr_auc,
                                                             best_epoch, args)

                    if early_stop.step(stopping_metric) and args.early_stop:
                        print('[INFO] Early Stopping')
                        break
                    epoch += 1

            wandb.log({"Best_Validation_TSS": best_tss,
                       "Best_Validation_epoch": best_epoch,
                       'Best_Validation_PR_AUC': best_pr_auc})

            # reload best tss checkpoint and test
            print("[INFO] Loading model at epoch:" + str(best_epoch))
            # noinspection PyBroadException
            try:
                model.load_state_dict(
                    torch.load(os.path.join(wandb.run.dir, 'model_tss.pt')))
            except:
                print('No model loaded... Loading default')
                # model.load_state_dict(torch.load(
                #     os.path.join('saved/models/TCN_4_3_8_0.91',
                #                  'model_tss.pt')))

            test_recall, test_precision, test_accuracy, test_bacc, \
            test_hss, test_tss = main_TCN_Liu.test(model, device, test_loader,
                                                   criterion, epoch)

            val_recall1list.append(val_recall)
            val_precision1list.append(val_precision)
            val_acclist.append(val_accuracy)
            val_bacclist.append(val_bacc)
            val_tsslist.append(val_tss)
            val_hsslist.append(val_hss)
            test_recall1list.append(test_recall)
            test_precision1list.append(test_precision)
            test_acclist.append(test_accuracy)
            test_bacclist.append(test_bacc)
            test_tsslist.append(test_tss)
            test_hsslist.append(test_hss)



    avg_recall0_list = []
    std_recall0_list = []
    avg_precision0_list = []
    std_precision0_list = []
    avg_acc_list = []
    std_acc_list = []
    avg_bacc_list = []
    std_bacc_list = []
    avg_hss_list = []
    std_hss_list = []
    avg_tss_list = []
    std_tss_list = []










