"""
Ecclesiastes 5:12 New King James Version (NKJV)
12 The sleep of a laboring man is sweet,
Whether he eats little or much;
But the abundance of the rich will not permit him to sleep.
"""
import yaml
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import class_weight
from sklearn.metrics import make_scorer, balanced_accuracy_score
import sklearn.metrics
import seaborn as sn

import sys
import os
import numpy as np
import argparse
import time

import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F

import skorch
from skorch import NeuralNetClassifier
from skorch.callbacks import *
from skorch.helper import predefined_split
from skorch.dataset import Dataset
from skorch import toy

from interpret.interpreter import get_shap
from skorch_tools import skorch_utils

from data_loader import CustomDataset
from data_loader import data_loader
from utils import early_stopping
from model.tcn import TemporalConvNet
from model import metric
from model import mlp
from interpret import interpreter
from utils import confusion_matrix_plot
from utils import pdf
from utils import visualize_CV

import wandb
from torchsummary import summary

import crossval_fold

'''
TCN with n residual blocks will have a receptive field of
1 + 2*(kernel_size-1)*(2^n-1)
'''


def preprocess_customdataset(x_val, y_val):
    datasets = CustomDataset.CustomDataset(x_val, y_val)

    return datasets


class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size,
                 dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels,
                                   kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.init_weights()

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x):
        y1 = self.tcn(x)  # input should have dimension (batch, chan, seq len)
        out = self.linear(y1[:, :, -1])
        return F.softmax(out, dim=1)


def train(model, device, train_loader, optimizer, epoch, criterion, args,
          nclass=2):
    start = time.time()
    model.train()
    confusion_matrix = torch.zeros(nclass, nclass)
    loss_epoch = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        # data = data.view(-1, cfg.n_features, cfg.seq_len)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss_epoch += criterion(output, target).item()
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(output.data, 1)

        for t, p in zip(target.view(-1), predicted.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1

    loss_epoch /= len(train_loader.dataset)
    recall, precision, accuracy, bacc, tss, hss, tp, fn, fp, tn, \
    mcc = metric.calculate_metrics(
        confusion_matrix.numpy(), nclass)

    # calculate predicted values
    yhat = infer_model(model, device, train_loader, args)

    # PR curves on train
    f1, pr_auc = metric.get_pr_auc(yhat, train_loader.dataset.targets)[2:4]
    # f1, pr_auc = 0, 0

    end = time.time()

    # print info
    print('{:<11s}{:^9d}{:^9.1f}{:^9.4f}'
          '{:^9.4f}{:^9.4f}{:^9.4f}{:^9.4f}'
          '{:^9.4f}{:^9.4f}'
          '{:^9.4f}{:^9.4f}{:^9.4f}'.format('Train', epoch, end - start, tss,
                                            pr_auc, hss, bacc, accuracy,
                                            precision, recall, f1, loss_epoch,
                                            mcc))

    wandb.log({"Training_Accuracy": accuracy, "Training_TSS": tss,
               "Training_HSS": hss, "Training_BACC": bacc,
               "Training_Precision": precision, "Training_Recall": recall,
               "Training_Loss": loss_epoch, "Training_F1": f1,
               "Training_PR_AUC": pr_auc, 'Train_MCC': mcc}, step=epoch)

    return recall, precision, accuracy, bacc, hss, tss


def validate(model, device, valid_loader, criterion, epoch, best_tss,
             best_pr_auc, best_epoch, args, nclass=2):
    start = time.time()
    model.eval()
    valid_loss = 0
    correct = 0
    confusion_matrix = torch.zeros(nclass, nclass)

    with torch.no_grad():
        for data, target in valid_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            valid_loss += criterion(output, target).item()
            # get the index of the max log-probability
            _, predicted = torch.max(output.data, 1)
            correct += predicted.eq(target.view_as(predicted)).sum().item()

            for t, p in zip(target.view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

    valid_loss /= len(valid_loader.dataset)

    # print("Validation Scores:")
    recall, precision, accuracy, bacc, tss, hss, tp, fn, fp, tn, \
    mcc = metric.calculate_metrics(
        confusion_matrix.numpy(), nclass)

    # calculate predicted values
    yhat = infer_model(model, device, valid_loader, args)

    # PR curves on train
    f1, pr_auc = metric.get_pr_auc(yhat, valid_loader.dataset.targets)[2:4]

    end = time.time()

    wandb.log({"Validation_Accuracy": accuracy, "Validation_TSS": tss,
               "Validation_HSS": hss, "Validation_BACC": bacc,
               "Validation_Precision": precision, "Validation_Recall": recall,
               "Validation_Loss": valid_loss, "Validation_F1": f1,
               "Validation_PR_AUC": pr_auc, "Validation_MCC": mcc}, step=epoch)

    # checkpoint on best metric
    cp = ''
    if tss >= best_tss:  # change to required metric
        best_pr_auc = pr_auc
        best_tss = tss
        best_epoch = epoch
        torch.save(model.state_dict(),
                   os.path.join(wandb.run.dir, 'model_tss.pt'))
        cp = '+'
    if pr_auc >= best_pr_auc:  # saves best auc model
        # best_pr_auc = pr_auc
        # best_tss = tss[0]
        # best_epoch = epoch
        torch.save(model.state_dict(),
                   os.path.join(wandb.run.dir, 'model_pr_auc.pt'))

    print('{:<11s}{:^9d}{:^9.1f}{:^9.4f}'
          '{:^9.4f}{:^9.4f}{:^9.4f}{:^9.4f}'
          '{:^9.4f}{:^9.4f}'
          '{:^9.4f}{:^9.4f}{:^9.4f}{:^3s}'.format('Valid', epoch, end - start,
                                                  tss, pr_auc, hss, bacc,
                                                  accuracy, precision, recall,
                                                  f1, valid_loss, mcc, cp))

    stopping_metric = best_tss
    return stopping_metric, best_tss, best_pr_auc, best_epoch, recall, \
           precision, accuracy, bacc, hss, tss


def test(model, device, test_loader, criterion, epoch, nclass=2):
    start = time.time()
    model.eval()
    test_loss = 0
    correct = 0
    confusion_matrix = torch.zeros(nclass, nclass)

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += criterion(output, target).item()
            # get the index of the max log-probability
            _, predicted = torch.max(output.data, 1)
            correct += predicted.eq(target.view_as(predicted)).sum().item()

            for t, p in zip(target.view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

    test_loss /= len(test_loader.dataset)

    print("Test Scores:")
    recall, precision, accuracy, bacc, tss, hss, tp, fn, fp, tn, \
    mcc = metric.calculate_metrics(
        confusion_matrix.numpy(), nclass)

    end = time.time()
    print('{:<11s}{:^9d}{:^9.1f}{:^9.4f}'
          '{:^9s}{:^9.4f}{:^9.4f}{:^9.4f}'
          '{:^9.4f}{:^9.4f}'
          '{:^9s}{:^9.4f}{:^9.4f}'.format('Test', epoch, end - start, tss, ' ',
                                          hss, bacc, accuracy, precision,
                                          recall, ' ', test_loss, mcc))

    wandb.log({"Test_Accuracy": accuracy, "Test_TSS": tss, "Test_HSS": hss,
               "Test_BACC": bacc, "Test_Precision": precision,
               "Test_Recall": recall, "Test_Loss": test_loss, "Test_MCC": mcc})

    return recall, precision, accuracy, bacc, hss, tss


def infer_model(model, device, data_loader, args):
    """    :param args:
         :return prediction of inferred data loader
    """
    model.eval()
    output_arr = []
    with torch.no_grad():
        # output = model(data_loader.dataset.data.to(device))
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            # data = data.view(-1, args.n_features, args.seq_len)
            output = model(data)
            output_arr.append(output.cpu().detach().numpy())

    return np.vstack(output_arr)


def init_project():
    with open('config-defaults.yaml', 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.Loader)

    model_type = cfg['model_type']['value']
    project = 'liu_pytorch_MLP' if model_type == 'MLP' else \
        'liu_pytorch_tcn'
    tags = cfg['tag']['value']

    return project, tags


if __name__ == '__main__':
    project, tags = init_project()
    wandb.init(project=project, tags=tags)
    cfg = wandb.config

    # initialize parameters
    filepath = './Data/Liu/' + cfg.flare_label + '/'
    # n_features = 0
    if cfg.flare_label == 'M5':
        n_features = cfg.n_features  # 20 original
    elif cfg.flare_label == 'M':
        n_features = 22
    elif cfg.flare_label == 'C':
        n_features = 14
    feature_names = data_loader.get_feature_names(
        filepath + 'normalized_training.csv')

    # initialize parameters
    start_feature = 5
    mask_value = 0
    nclass = 2
    num_of_fold = 10

    # GPU check
    use_cuda = cfg.cuda and torch.cuda.is_available()
    if cfg.cuda and torch.cuda.is_available():
        print("Cuda enabled and available")
    elif cfg.cuda and not torch.cuda.is_available():
        print("Cuda enabled not not available, CPU used.")
    elif not cfg.cuda:
        print("Cuda disabled")
    device = torch.device("cuda" if use_cuda else "cpu")

    # set seed
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    np.random.seed(cfg.seed)

    # setup dataloaders
    X_train_data, y_train_data = data_loader.load_data(
        datafile=filepath + 'normalized_training.csv',
        flare_label=cfg.flare_label, series_len=cfg.seq_len,
        start_feature=start_feature, n_features=cfg.n_features,
        mask_value=mask_value)
    X_train_fold, y_train_fold = data_loader.partition_10_folds(X_train_data,
                                                                y_train_data,
                                                                num_of_fold)

    X_valid_data, y_valid_data = data_loader.load_data(
        datafile=filepath + 'normalized_validation.csv',
        flare_label=cfg.flare_label, series_len=cfg.seq_len,
        start_feature=start_feature, n_features=cfg.n_features,
        mask_value=mask_value)
    X_valid_fold, y_valid_fold = data_loader.partition_10_folds(X_valid_data,
                                                                y_valid_data,
                                                                num_of_fold)

    X_test_data, y_test_data = data_loader.load_data(
        datafile=filepath + 'normalized_testing.csv',
        flare_label=cfg.flare_label, series_len=cfg.seq_len,
        start_feature=start_feature, n_features=cfg.n_features,
        mask_value=mask_value)
    X_test_fold, y_test_fold = data_loader.partition_10_folds(X_test_data,
                                                              y_test_data,
                                                              num_of_fold)

    # crossval_fold.cross_val_train(num_of_fold, X_train_fold, y_train_fold,
    #                               X_valid_fold, y_valid_fold, X_test_fold,
    #                               y_test_fold, cfg, nclass, device)

    y_train_tr = data_loader.label_transform(y_train_data)
    y_valid_tr = data_loader.label_transform(y_valid_data)
    y_test_tr = data_loader.label_transform(y_test_data)

    if cfg.model_type == 'MLP':
        X_train_data = np.reshape(X_train_data,
                                  (len(X_train_data), cfg.n_features))
        X_valid_data = np.reshape(X_valid_data,
                                  (len(X_valid_data), cfg.n_features))
        X_test_data = np.reshape(X_test_data,
                                  (len(X_test_data), cfg.n_features))
    elif cfg.model_type == 'TCN':
        X_train_data = torch.tensor(X_train_data).float()
        X_train_data = X_train_data.permute(0, 2, 1)
        X_valid_data = torch.tensor(X_valid_data).float()
        X_valid_data = X_valid_data.permute(0, 2, 1)
        X_test_data = torch.tensor(X_test_data).float()
        X_test_data = X_test_data.permute(0, 2, 1)

    # (samples, seq_len, features) -> (samples, features, seq_len)
    X_train_data_tensor = torch.tensor(X_train_data).float()
    y_train_tr_tensor = torch.tensor(y_train_tr).long()

    X_valid_data_tensor = torch.tensor(X_valid_data).float()
    y_valid_tr_tensor = torch.tensor(y_valid_tr).long()

    X_test_data_tensor = torch.tensor(X_test_data).float()
    y_test_tr_tensor = torch.tensor(y_test_tr).long()

    # ready custom dataset
    datasets = {'train': preprocess_customdataset(X_train_data_tensor,
                                                  y_train_tr_tensor),
                'valid': preprocess_customdataset(X_valid_data_tensor,
                                                  y_valid_tr_tensor),
                'test': preprocess_customdataset(X_test_data_tensor,
                                                 y_test_tr_tensor)}

    kwargs = {'num_workers': cfg.num_workers,
              'pin_memory': True} if use_cuda else {}

    train_loader = torch.utils.data.DataLoader(datasets['train'],
                                               cfg.batch_size, shuffle=False,
                                               drop_last=False, **kwargs)
    valid_loader = torch.utils.data.DataLoader(datasets['valid'],
                                               cfg.batch_size, shuffle=False,
                                               drop_last=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(datasets['test'], cfg.batch_size,
                                              shuffle=False, drop_last=False,
                                              **kwargs)
    # Shape: (batch size, features, seq_len)
    # make model
    channel_sizes = [cfg.nhid] * cfg.levels
    kernel_size = cfg.ksize
    dropout = cfg.dropout

    if cfg.model_type == 'MLP':
        model = mlp.MLPModule(input_units=cfg.n_features,
                              hidden_units=cfg.hidden_units,
                              num_hidden=cfg.layers,
                              dropout=cfg.dropout).to(device)
    elif cfg.model_type == "TCN":
        model = TCN(cfg.n_features, nclass, channel_sizes,
                    kernel_size=kernel_size, dropout=dropout).to(device)
        summary(model, input_size=(cfg.n_features, cfg.seq_len))

    wandb.watch(model, log='all')

    # optimizers
    class_weights = class_weight.compute_class_weight('balanced',
                                                      np.unique(y_train_data),
                                                      y_train_data)

    # noinspection PyArgumentList
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(
        device))  # weighted cross entropy
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate,
                                 weight_decay=cfg.weight_decay, amsgrad=False)

    # print model parameters
    print("Receptive Field: " + str(
        1 + 2 * (cfg.ksize - 1) * (2 ** cfg.levels - 1)))
    # print(len(list(model.parameters())))
    # for i in range(len(list(model.parameters()))):
    #     print(list(model.parameters())[i].size())

    # early stopping check
    early_stop = early_stopping.EarlyStopping(mode='max',
                                              patience=cfg.patience)
    best_tss = 0.0
    best_pr_auc = 0.0
    best_epoch = 0
    epoch = 0

    if not cfg.skorch:
        print('{:<11s}{:^9s}{:^9s}{:^9s}'
              '{:^9s}{:^9s}{:^9s}{:^9s}'
              '{:^9s}{:^9s}'
              '{:^9s}{:^9s}{:^9s}{:^3s}'.format('Data Loader', 'Epoch',
                                                'Runtime', 'TSS', 'PR_AUC',
                                                'HSS', 'BACC', 'ACC',
                                                'Precision', 'Recall', 'F1',
                                                'Loss', 'MCC', 'CP'))

        if cfg.training:
            while epoch < cfg.epochs:
                train_tss = train(model, device, train_loader, optimizer, epoch,
                                  criterion, cfg)[5]
                stopping_metric, best_tss, best_pr_auc, best_epoch = validate(
                    model, device, valid_loader, criterion, epoch, best_tss,
                    best_pr_auc, best_epoch, cfg)[0:4]

                if early_stop.step(stopping_metric) and cfg.early_stop:
                    print('[INFO] Early Stopping')
                    break

                # # Continue training if recently improved
                # if epoch == cfg.epochs-1 and early_stop.num_bad_epochs < 2:
                #     cfg.epochs += 5
                #     print("[INFO] not finished training...")
                epoch += 1

        wandb.log(
            {"Best_Validation_TSS": best_tss, "Best_Validation_epoch":
            best_epoch,
             'Best_Validation_PR_AUC': best_pr_auc})

        # reload best tss checkpoint and test
        print("[INFO] Loading model at epoch:" + str(best_epoch))
        # noinspection PyBroadException
        try:
            model.load_state_dict(
                torch.load(os.path.join(wandb.run.dir, 'model_tss.pt')))
        except:
            print('No model loaded... Loading default')
            weights_file = wandb.restore('model.pt',
                                       run_path="dewald123/liu_pytorch_tcn/3tcj8ahy")
            model.load_state_dict(torch.load(weights_file.name))

        test_tss = test(model, device, test_loader, criterion, epoch)[5]

        '''
        PR Curves
        '''
        # Train
        yhat = infer_model(model, device, train_loader, cfg)

        f1, pr_auc = metric.plot_precision_recall(model, yhat, y_train_tr_tensor,
                                                 'Train')[2:4]
        metric.plot_confusion_matrix(yhat, y_train_tr_tensor, 'Train')
        tss = metric.get_metrics_threshold(yhat, y_train_tr_tensor)[4]

        # Validation
        yhat = infer_model(model, device, valid_loader, cfg)

        f1, pr_auc = metric.plot_precision_recall(model, yhat, y_valid_tr_tensor,
                                           'Validation')[2:4]
        metric.plot_confusion_matrix(yhat, y_valid_tr_tensor, 'Validation')
        tss = metric.get_metrics_threshold(yhat, y_valid_tr_tensor)[4]
        th = metric.get_metrics_threshold(yhat, y_valid_tr_tensor)[10]
        roc_auc = metric.get_roc(model, yhat, y_valid_tr_tensor, device,
                                 'Validation')
        th_norm = pdf.plot_density_estimation(yhat, y_valid_tr_tensor,
                                              'Validation')

        # Test
        yhat = infer_model(model, device, test_loader, cfg)
        cm = sklearn.metrics.confusion_matrix(y_test_tr_tensor,
                                              metric.to_labels(yhat[:, 1],
                                                               th))  # watch
        tss_th = metric.calculate_metrics(cm, 2)[4]

        f1, pr_auc = metric.plot_precision_recall(model, yhat,
        y_test_tr_tensor, 'Test')[2:4]
        metric.plot_confusion_matrix(yhat, y_test_tr_tensor, 'Test')
        tss = metric.get_metrics_threshold(yhat, y_test_tr_tensor)[4]

        roc_auc = metric.get_roc(model, yhat, y_test_tr_tensor, device, 'Test')

        print('Test TSS from validation threshold ({:0.3f}): {:0.3f}'.format(th,
                                                                        tss_th))
        wandb.log({'Test_TSS_Th': tss_th})

        th_norm_test = pdf.plot_density_estimation(yhat, y_test_tr_tensor,
        'Test')

    '''
    Model interpretation
    '''
    # todo interpret on test set?
    #
    # test_loader_interpret = torch.utils.data.DataLoader(datasets['test'],
    # int(
    #     cfg.batch_size / 6), shuffle=False, drop_last=False)
    #
    # attr_ig, attr_sal, attr_ig_avg, attr_sal_avg =
    # interpreter.interpret_model(
    #     model, device, test_loader_interpret, cfg.n_features, cfg)
    #
    # interpreter.visualize_importance(
    #     np.array(feature_names[start_feature:start_feature +
    #     cfg.n_features]),
    #     np.mean(attr_ig_avg, axis=0), np.std(attr_ig_avg, axis=0),
    #     cfg.n_features,
    #     title="Integrated Gradient Features")
    #
    # interpreter.visualize_importance(
    #     np.array(feature_names[start_feature:start_feature +
    #     cfg.n_features]),
    #     np.mean(attr_sal_avg, axis=0), np.std(attr_sal_avg, axis=0),
    #     cfg.n_features, title="Saliency Features")
    #
    # '''SHAP'''
    # plt.close('all')
    # get_shap(model, test_loader, device, cfg, feature_names, start_feature)

    if cfg.skorch:
        '''
            Skorch training
        '''
        # correct format for skorch
        test_inputs = X_test_data_tensor.numpy()
        test_labels = y_test_tr_tensor.numpy()
        inputs = X_train_data_tensor.numpy()
        labels = y_train_tr_tensor.numpy()

        X_valid_data = X_valid_data_tensor.numpy()
        y_valid_tr = y_valid_tr_tensor.numpy()
        valid_ds = Dataset(X_valid_data, y_valid_tr)

        # combined datasets
        combined_inputs = np.concatenate([inputs, X_valid_data], axis=0).astype(np.float32)
        combined_labels = np.concatenate([labels, y_valid_tr], axis=0)
        ds = Dataset(combined_inputs, combined_labels)
        combined_labels = np.array([labels for _, labels in iter(ds)])

        # Metrics + Callbacks
        valid_tss = EpochScoring(scoring=make_scorer(skorch_utils.get_tss),
                                 lower_is_better=False, name='valid_tss',
                                 use_caching=True)
        train_tss = EpochScoring(scoring=make_scorer(skorch_utils.get_tss),
                                 lower_is_better=False, name='train_tss',
                                 use_caching=True, on_train=True)
        valid_hss = EpochScoring(scoring=make_scorer(skorch_utils.get_hss),
                                 lower_is_better=False, name='valid_hss',
                                 use_caching=True)
        train_bacc = EpochScoring(
            scoring=make_scorer(balanced_accuracy_score, **{'adjusted': True}),
            lower_is_better=False, name='train_bacc', use_caching=True,
            on_train=True)

        if cfg.early_stop:
            earlystop = EarlyStopping(monitor='valid_tss', lower_is_better=False,
                                      patience=cfg.patience)
        else:
            earlystop = None
        savename = '{}_{}_{}_{}_{}_{}'.format(cfg.model_type, cfg.layers,
                                        cfg.hidden_units, cfg.batch_size,
                                              cfg.learning_rate, cfg.seed)
        checkpoint = Checkpoint(monitor='valid_tss_best',
                                dirname='./saved/models/' + savename)

        logger = skorch_utils.LoggingCallback(test_inputs, test_labels)

        load_state = LoadInitState(checkpoint)
        reload_at_end = skorch_utils.LoadBestCP(checkpoint)

        # noinspection PyArgumentList
        net = NeuralNetClassifier(model, max_epochs=cfg.epochs,
                                  batch_size=cfg.batch_size,
                                  criterion=nn.CrossEntropyLoss,
                                  criterion__weight=torch.FloatTensor(
                                      class_weights).to(device),
                                  optimizer=torch.optim.Adam,
                                  optimizer__lr=cfg.learning_rate,
                                  optimizer__weight_decay=cfg.weight_decay,
                                  optimizer__amsgrad=False, device=device,
                                  # train_split=skorch.dataset.CVSplit(
                                  #     cv=cfg.n_splits, stratified=False),
                                  train_split=predefined_split(valid_ds),
                                  # train_split=None,
                                  callbacks=[train_tss, valid_tss, earlystop,
                                             checkpoint,  # load_state,
                                             reload_at_end,
                                             logger],
                                  # iterator_train__shuffle=True,
                                  warm_start=False)

        net.initialize()
        init_savename = 'init_{}_{}'.format(cfg.model_type, cfg.seed)
        net.save_params(f_params=init_savename)

        if not cfg.cross_validation:
            net.fit(inputs, labels)

        # '''
        # Cross Validation
        # '''
        elif cfg.cross_validation:
            kf = sklearn.model_selection.KFold(n_splits=cfg.n_splits, shuffle=False)
            skf = sklearn.model_selection.StratifiedKFold(cfg.n_splits, shuffle=False)
            tscv = sklearn.model_selection.TimeSeriesSplit(n_splits=cfg.n_splits)
            rkf = sklearn.model_selection.RepeatedKFold(n_splits=5,
                                                        n_repeats=2)
            scores = []
            visualize_CV.visualize_cv(sklearn.model_selection.RepeatedKFold,
                                      combined_inputs, combined_labels, cfg)
            for train_index, val_index in rkf.split(combined_inputs,
                                                  combined_labels):
                print('train -  {}   |   test -  {}'.format(
                    np.bincount(combined_labels[train_index]), np.bincount(combined_labels[val_index])))
                x_train, x_val = combined_inputs[train_index], combined_inputs[val_index]
                y_train, y_val = combined_labels[train_index], combined_labels[val_index]
                net.train_split = predefined_split(Dataset(x_val, y_val))
                net.load_params(f_params=init_savename)  # Reload inital params
                net.fit(x_train, y_train)
                predictions = net.predict(x_val)
                scores.append(
                    balanced_accuracy_score(y_val, predictions, adjusted=True))
            print('Scores from each Iteration: ', scores)
            print('Average K-Fold Score :', np.mean(scores))
            wandb.log({"Avg_CV_Score": np.mean(scores)})

        '''
        Test Results
        '''
        # train on train and val set
        if cfg.cross_validation:
            net.initialize()
            net.load_params(f_params=init_savename)
            net.train_split = None
            net.callbacks = [train_tss, Checkpoint(monitor='train_tss_best',
                                                   dirname='./saved/models/' + savename)]
            net.initialize_callbacks()
            net.fit_loop(combined_inputs, combined_labels, epochs=100)

        net.initialize()
        net.load_params(checkpoint=checkpoint)  # Select best TSS epoch

        y_test = net.predict(test_inputs)
        tss_test_score = skorch_utils.get_tss(test_labels, y_test)
        wandb.log({'Test_TSS': tss_test_score})
        print("Test TSS:" + str(tss_test_score))

    # Save model to W&B
    torch.save(model.state_dict(), os.path.join(wandb.run.dir, 'model.pt'))
    print('Finished')
