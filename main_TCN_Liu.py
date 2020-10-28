"""
Ecclesiastes 5:12 New King James Version (NKJV)
12 The sleep of a laboring man is sweet,
Whether he eats little or much;
But the abundance of the rich will not permit him to sleep.
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import *
from torch.autograd import gradcheck
import wandb
import yaml
import sklearn
from sklearn.metrics import make_scorer, balanced_accuracy_score
from sklearn.model_selection import GridSearchCV, cross_val_score, \
    StratifiedKFold, KFold
from sklearn.utils import class_weight
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import skorch
from skorch import NeuralNetClassifier
from skorch.callbacks import *
from skorch.dataset import Dataset
from skorch.helper import predefined_split
from torchsummary import summary
import matplotlib.pyplot as plt
from torch_lr_finder import LRFinder
from wandb import sklearn

from interpret import interpreter
import crossval_fold
from data_loader import CustomDataset
from data_loader import data_loader
from model import lstm
from model import metric
from model import mlp
from model import tcn
from model.tcn import TemporalConvNet
from skorch_tools import skorch_utils
from utils import early_stopping
from utils import pdf
from utils import visualize_CV
from utils import lr_finding

'''
TCN with n residual blocks will have a receptive field of
1 + 2*(kernel_size-1)*(2^n-1)
'''


def preprocess_customdataset(x_val, y_val):
    datasets = CustomDataset.CustomDataset(x_val, y_val)

    return datasets


class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size,
                 dropout, attention=False):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels,
                                   kernel_size=kernel_size, dropout=dropout,
                                   attention=attention)
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.init_weights()

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x):
        y1 = self.tcn(x)  # input should have dimension (batch, chan, seq len)
        out = self.linear(y1[:, :, -1])
        return out


def train(model, device, train_loader, optimizer, epoch, criterion, cfg,
          scheduler=None, nclass=2):
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
        if cfg.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip)
        optimizer.step()
        _, predicted = torch.max(output.data, 1)

        for t, p in zip(target.view(-1), predicted.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1

        if scheduler is not None:
            scheduler.step()
        else:
            pass

    loss_epoch /= len(train_loader.dataset)
    recall, precision, accuracy, bacc, tss, hss, tp, fn, fp, tn, \
    mcc = metric.calculate_metrics(
        confusion_matrix.numpy(), nclass)

    # calculate predicted values
    yhat = infer_model(model, device, train_loader)

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
               "Training_PR_AUC": pr_auc, 'Train_MCC': mcc, 'Epoch':epoch},
              step=epoch)

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
    yhat = infer_model(model, device, valid_loader)

    # PR curves on train
    f1, pr_auc = metric.get_pr_auc(yhat, valid_loader.dataset.targets)[2:4]

    end = time.time()

    wandb.log({"Validation_Accuracy": accuracy, "Validation_TSS": tss,
               "Validation_HSS": hss, "Validation_BACC": bacc,
               "Validation_Precision": precision, "Validation_Recall": recall,
               "Validation_Loss": valid_loss, "Validation_F1": f1,
               "Validation_PR_AUC": pr_auc, "Validation_MCC": mcc,'Epoch':epoch},
              step=epoch)

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
        # best_val_tss = tss[0]
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

    # print("Test Scores:")
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

    wandb.log({"Test_Accuracy_curve": accuracy, "Test_TSS_curve": tss,
               "Test_HSS_curve": hss, "Test_BACC_curve": bacc,
               "Test_Precision_curve": precision, "Test_Recall_curve": recall,
               "Test_Loss_curve": test_loss, "Test_MCC_curve": mcc,'Epoch':epoch})

    return recall, precision, accuracy, bacc, hss, tss


def infer_model(model, device, data_loader):
    """    :param args:
         :return prediction of inferred data loader
    """
    model.eval()
    output_arr = []
    with torch.no_grad():
        # output = model(data_loader.dataset.data.to(device))
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            # data = data.view(-1, cfg.n_features, cfg.seq_len)
            output = model(data)
            output_arr.append(output.cpu().detach().numpy())

    return np.vstack(output_arr)


def do_nested_cv():
    # weight_norm in cnn breaks this
    p_grid = {}
    net.initialize()
    net.load_params(f_params=init_savename)
    net.train_split = None
    net.callbacks = [train_tss_cb, lrscheduler]
    net.initialize_callbacks()

    inner_cv = KFold(n_splits=2, shuffle=True, random_state=cfg.seed)
    outer_cv = KFold(n_splits=5, shuffle=True, random_state=cfg.seed)

    gcv = GridSearchCV(estimator=net, param_grid=p_grid,
                       scoring=make_scorer(balanced_accuracy_score,
                                           **{'adjusted': True}),
                       cv=inner_cv, refit=True, return_train_score=True,
                       verbose=1)

    nested_score = cross_val_score(gcv, X=combined_inputs, y=combined_labels,
                                   cv=outer_cv,
                                   scoring=make_scorer(balanced_accuracy_score,
                                                       **{'adjusted': True}))

    print('%s | outer TSS %.4f +/- %.4f' % (
        'Model', nested_score.mean(), nested_score.std()))

    # Fitting a model to the whole training set
    # using the "best" algorithm
    # best_algo = gridcvs['SVM']
    gcv.fit(combined_inputs, combined_labels)

    # summarize results
    print("Best: %f using %s" % (gcv.best_score_, gcv.best_params_))
    means = gcv.cv_results_['mean_test_score']
    stds = gcv.cv_results_['std_test_score']
    params = gcv.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%.4f (%.4f) with: %r" % (mean, stdev, param))

    train_tss = balanced_accuracy_score(y_true=combined_labels,
                                        y_pred=gcv.predict(combined_inputs),
                                        adjusted=True)
    print('Training TSS: %.4f' % (train_tss))

    # save to csv
    nested_score_df = pd.Series(nested_score, name='best_outer_score')
    df_csv = pd.concat([pd.DataFrame(gcv.cv_results_), nested_score_df],
                       axis=1)
    # df_csv.to_csv(
    #     '../saved/scores/nestedcv_{}_{}.csv'.format(seed, cfg.model_type))

    return gcv


def parse_args(cfg):
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=cfg.seed, metavar='N',
                        help='init seed')
    parser.add_argument('--batch_size', type=int, default=cfg.batch_size,
                        metavar='N',
                        help='mini-batch size')
    parser.add_argument('--optim', type=str, default=cfg.optim, metavar='N',
                        help='optimizer SGD/Adam')
    parser.add_argument('--dataset', type=str, default=cfg.dataset,
                        metavar='N',
                        help='(Liu/Liu_train/Liu_z/Krynauw)')
    parser.add_argument('--model_type', type=str, default=cfg.model_type,
                        metavar='N',
                        help='CNN/TCN/MLP')
    parser.add_argument('--layers', type=int, default=cfg.layers, metavar='N',
                        help='MLP layers')
    parser.add_argument('--levels', type=int, default=cfg.levels, metavar='N',
                        help='CNN/TCN layers')
    parser.add_argument('--hidden_units', type=int, default=cfg.hidden_units,
                        metavar='N',
                        help='MLP nodes per layers')
    parser.add_argument('--ksize', type=int, default=cfg.ksize, metavar='N',
                        help='kernel size')
    parser.add_argument('--nhid', type=int, default=cfg.nhid, metavar='N',
                        help='number of filters')
    parser.add_argument('--dropout', type=float, default=cfg.dropout,
                        metavar='N',
                        help='dropout amount')
    parser.add_argument('--weight_decay', type=float, default=cfg.weight_decay,
                        help='how much weight decay')
    parser.add_argument('--lr_rangetest_iter', type=int,
                        default=cfg.lr_rangetest_iter,
                        help='how many iterations')

    args = parser.parse_args()
    wandb.config.update(args, allow_val_change=True)  # adds all of the arguments as config


def init_project():
    with open('config-defaults.yaml', 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.Loader)

    model_type = cfg['model_type']['value']
    if model_type == 'MLP':
        project = 'liu_pytorch_MLP'
    elif model_type == 'TCN':
        project = 'liu_pytorch_tcn'
    elif model_type == 'CNN':
        project = 'liu_pytorch_cnn'
    elif model_type == 'RNN':
        project = 'liu_pytorch_lstm'
    tags = cfg['tag']['value']

    return project, tags


if __name__ == '__main__':
    project, tags = init_project()
    run = wandb.init(project=project, tags=tags)
    cfg = wandb.config

    # Parse args
    if cfg.parse_args:
        parse_args(cfg)

    # initialize parameters
    filepath = './Data/' + cfg.dataset
    # artifact = wandb.Artifact('liu-dataset', type='dataset')
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

    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    np.random.seed(cfg.seed)

    # feature select list
    sharps = ['USFLUX', 'SAVNCPP', 'TOTPOT', 'ABSNJZH', 'SHRGT45', 'AREA_ACR',
              'R_VALUE', 'TOTUSJH', 'TOTUSJZ', 'MEANJZH', 'MEANJZD', 'MEANPOT',
              'MEANSHR', 'MEANALP', 'MEANGAM', 'MEANGBZ', 'MEANGBT',
              'MEANGBH', ]  # 18
    lorentz = ['TOTBSQ', 'TOTFX', 'TOTFY', 'TOTFZ', 'EPSX', 'EPSY',
               'EPSZ']  # 7
    history_features = ['Bdec', 'Cdec', 'Mdec', 'Xdec', 'Edec', 'logEdec',
                        'Bhis', 'Chis', 'Mhis', 'Xhis', 'Bhis1d', 'Chis1d',
                        'Mhis1d', 'Xhis1d', 'Xmax1d']  # 15
    listofuncorrfeatures = ['TOTUSJH', 'SAVNCPP', 'ABSNJZH', 'TOTPOT',
                            'AREA_ACR', 'Cdec', 'Chis', 'Edec', 'Mhis',
                            'Xmax1d', 'Mdec', 'MEANPOT', 'R_VALUE', 'Mhis1d',
                            'MEANGAM', 'TOTFX', 'MEANJZH', 'MEANGBZ', 'TOTFZ',
                            'TOTFY', 'logEdec', 'EPSZ', 'MEANGBH', 'MEANJZD',
                            'Xhis1d', 'Xdec', 'Xhis', 'EPSX', 'EPSY', 'Bhis',
                            'Bdec', 'Bhis1d'] # 32
    all_f = sharps+lorentz+history_features
    bad_features = ['MEANPOT', 'Mhis1d', 'Edec', 'Xhis1d', 'Bdec','Bhis',
                    'Bhis1d']
    # feature_list = [x for x in all if x not in bad_features] #
    # feature_list = feature_names[5:]
    feature_list = all_f 
    # can be
    # None, need to change
    # cfg.n_features to match length

    # get receptive field
    if cfg.model_type == "TCN":
        receptive_field = 1 + 2 * (cfg.ksize - 1) * (2 ** cfg.levels - 1)
        wandb.config.update({"seq_len": receptive_field},
                            allow_val_change=True)
        print("Receptive Field: " + str(receptive_field))
    elif cfg.model_type == "CNN":
        receptive_field = cfg.ksize
        wandb.config.update({"seq_len": cfg.ksize}, allow_val_change=True)
        print("Receptive Field: " + str(receptive_field))

    # setup dataloaders
    if (cfg.dataset != 'Synth/') and (cfg.dataset != 'Sampled/'):
        X_train_data, y_train_data = data_loader.load_data(
            datafile=filepath + 'normalized_training.csv',
            flare_label=cfg.flare_label, series_len=cfg.seq_len,
            start_feature=start_feature, n_features=cfg.n_features,
            mask_value=mask_value, feature_list=feature_list)
        X_train_fold, y_train_fold = data_loader.partition_10_folds(X_train_data,
                                                                    y_train_data,
                                                                    num_of_fold)

        X_valid_data, y_valid_data = data_loader.load_data(
            datafile=filepath + 'normalized_validation.csv',
            flare_label=cfg.flare_label, series_len=cfg.seq_len,
            start_feature=start_feature, n_features=cfg.n_features,
            mask_value=mask_value, feature_list=feature_list)
        X_valid_fold, y_valid_fold = data_loader.partition_10_folds(X_valid_data,
                                                                    y_valid_data,
                                                                    num_of_fold)

        X_test_data, y_test_data = data_loader.load_data(
            datafile=filepath + 'normalized_testing.csv',
            flare_label=cfg.flare_label, series_len=cfg.seq_len,
            start_feature=start_feature, n_features=cfg.n_features,
            mask_value=mask_value, feature_list=feature_list)
        X_test_fold, y_test_fold = data_loader.partition_10_folds(X_test_data,
                                                                  y_test_data,
                                                                  num_of_fold)

        if cfg.liu_fold:
            crossval_fold.cross_val_train(num_of_fold, X_train_fold, y_train_fold,
                                          X_valid_fold, y_valid_fold, X_test_fold,
                                          y_test_fold, cfg, nclass, device)

        y_train_tr = data_loader.label_transform(y_train_data)
        y_valid_tr = data_loader.label_transform(y_valid_data)
        y_test_tr = data_loader.label_transform(y_test_data)
    elif cfg.dataset == 'Synth/':
        print('Synth dataset')
        X, y = make_classification(n_samples=100000, n_features=cfg.n_features,
                                   n_redundant=0, n_informative=1,
                                   n_clusters_per_class=1,
                                   weights=[0.99, 0.01], class_sep=1,
                                   random_state=cfg.seed)

        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=0.33,
                                                            random_state=cfg.seed)
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train,
                                                              test_size=0.33,
                                                              random_state=cfg.seed)
        X_train_data = X_train.astype(np.float32)
        X_valid_data = X_valid.astype(np.float32)
        X_test_data = X_test.astype(np.float32)
        y_train_tr = y_train.astype(np.int64)
        y_valid_tr = y_valid.astype(np.int64)
        y_test_tr = y_test.astype(np.int64)
    elif cfg.dataset == 'Sampled/':
        df_train = pd.read_csv(filepath + 'normalized_training.csv')
        X_train = df_train.iloc[:,1:].to_numpy()
        y_train = df_train.iloc[:,0].to_numpy()

        print(X_train.shape)

        X_valid_data, y_valid_data = data_loader.load_data(
            datafile=filepath + 'normalized_validation.csv',
            flare_label=cfg.flare_label, series_len=cfg.seq_len,
            start_feature=start_feature, n_features=cfg.n_features,
            mask_value=mask_value, feature_list=feature_list)

        X_test_data, y_test_data = data_loader.load_data(
            datafile=filepath + 'normalized_testing.csv',
            flare_label=cfg.flare_label, series_len=cfg.seq_len,
            start_feature=start_feature, n_features=cfg.n_features,
            mask_value=mask_value, feature_list=feature_list)

        y_valid_tr = data_loader.label_transform(y_valid_data)
        y_test_tr = data_loader.label_transform(y_test_data)

        X_train_data = X_train.astype(np.float32)
        y_train_tr = y_train.astype(np.int64)



    if cfg.model_type == 'MLP':
        X_train_data = np.reshape(X_train_data,
                                  (len(X_train_data), cfg.n_features))
        X_train_data_tensor = torch.tensor(X_train_data).float()
        X_valid_data = np.reshape(X_valid_data,
                                  (len(X_valid_data), cfg.n_features))
        X_valid_data_tensor = torch.tensor(X_valid_data).float()
        X_test_data = np.reshape(X_test_data,
                                 (len(X_test_data), cfg.n_features))
        X_test_data_tensor = torch.tensor(X_test_data).float()
    elif (cfg.model_type == 'TCN') or (cfg.model_type == 'CNN'):
        X_train_data = torch.tensor(X_train_data).float()
        X_train_data_tensor = X_train_data.permute(0, 2, 1)
        X_valid_data = torch.tensor(X_valid_data).float()
        X_valid_data_tensor = X_valid_data.permute(0, 2, 1)
        X_test_data = torch.tensor(X_test_data).float()
        X_test_data_tensor = X_test_data.permute(0, 2, 1)
    else:
        X_train_data_tensor = torch.tensor(X_train_data).float()
        X_valid_data_tensor = torch.tensor(X_valid_data).float()
        X_test_data_tensor = torch.tensor(X_test_data).float()

    # (samples, seq_len, features) -> (samples, features, seq_len)
    y_train_tr_tensor = torch.tensor(y_train_tr).long()
    y_valid_tr_tensor = torch.tensor(y_valid_tr).long()
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
                                               cfg.batch_size,
                                               shuffle=True if cfg.shuffle else False,
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

    # Create model
    if cfg.model_type == 'MLP':
        model = mlp.MLPModule(input_units=cfg.n_features,
                              hidden_units=cfg.hidden_units,
                              num_hidden=cfg.layers,
                              dropout=cfg.dropout).to(device)
        # model = mlp.DEFNR(input_units=cfg.n_features,
        #                       hidden_units=cfg.hidden_units,
        #                       num_hidden=cfg.layers,
        #                       dropout=cfg.dropout).to(device)
        summary(model, input_size=(cfg.n_features, ))
    elif cfg.model_type == "TCN":
        print("Receptive Field: " + str(
            1 + 2 * (cfg.ksize - 1) * (2 ** cfg.levels - 1)))
        wandb.config.update({"seq_len": 1 + 2 * (cfg.ksize - 1) * (2 ** cfg.levels - 1)})
        model = TCN(cfg.n_features, nclass, channel_sizes,
                    kernel_size=kernel_size, dropout=cfg.dropout,
                    attention=False).to(device)
        summary(model, input_size=(cfg.n_features, cfg.seq_len))
    elif cfg.model_type == "CNN":
        wandb.config.update({"seq_len": cfg.ksize}, allow_val_change=True)
        model = tcn.Simple1DConv(cfg.n_features, cfg.nhid, cfg.levels,
                                 kernel_size=kernel_size, dropout=cfg.dropout).to(device)
        summary(model, input_size=(cfg.n_features, cfg.seq_len))
        print(f'Receptive field: {cfg.seq_len}')
    elif cfg.model_type == 'RNN':
        model = lstm.LSTMModel(cfg.n_features, cfg.nhid, cfg.levels,
                               output_dim=nclass, dropout=cfg.dropout, device=device,
                               rnn_module='LSTM')
        # summary(model, input_size=(cfg.seq_len,
        #                            cfg.n_features))

    # model = model.float() # too slow, approx is close enough
    wandb.watch(model, log='all')

    # optimizers
    class_weights = class_weight.compute_class_weight('balanced',
                                                      np.unique(y_train_tr),
                                                      y_train_tr)

    # noinspection PyArgumentList
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(
        device))  # weighted cross entropy
    if cfg.optim == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.learning_rate,
                                    weight_decay=cfg.weight_decay,
                                    nesterov=True, momentum=cfg.momentum)
        # find LR
        if cfg.lr_finder:
            min_lr, halfway_lr, max_lr = lr_finding.find_lr(model, optimizer,
                                                            criterion, device,
                                                            train_loader,
                                                            valid_loader, cfg)
            wandb.config.update(
                {"min_lr": min_lr, "learning_rate": halfway_lr, "max_lr":
                    max_lr},
                allow_val_change=True)

        if cfg.lr_scheduler:
            # scheduler = lr_scheduler.CyclicLR(optimizer,
            #                                   base_lr=cfg.min_lr,
            #                                   max_lr=cfg.max_lr,
            #                                   step_size_up=int(4 * (len(
            #                                       X_train_data) / cfg.batch_size)))
            scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=cfg.max_lr,
                                                steps_per_epoch=len(
                                                    train_loader),
                                                epochs=cfg.epochs)
    elif cfg.optim == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate,
                                     weight_decay=cfg.weight_decay)

        # find LR
        if cfg.lr_finder:
            min_lr, halfway_lr, max_lr = lr_finding.find_lr(model, optimizer,
                                                            criterion, device,
                                                            train_loader,
                                                            valid_loader, cfg)
            wandb.config.update({"min_lr": min_lr, "learning_rate": halfway_lr,
                                 "max_lr": max_lr}, allow_val_change=True)

        if cfg.lr_scheduler:
            # scheduler = lr_scheduler.CyclicLR(optimizer,
            #                                   base_lr=cfg.min_lr,
            #                                   max_lr=cfg.max_lr,
            #                                   step_size_up=int(4 * (len(
            #                                       X_train_data) / cfg.batch_size)),
            #                                   cycle_momentum=False)
            scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=cfg.max_lr,
                                                steps_per_epoch=len(
                                                    train_loader),
                                                epochs=cfg.epochs,
                                                cycle_momentum=False)


    # print model parameters

    # print(len(list(model.parameters())))
    # for i in range(len(list(model.parameters()))):
    #     print(list(model.parameters())[i].size())

    # early stopping check
    early_stop = early_stopping.EarlyStopping(mode='max',
                                              patience=cfg.patience)
    best_val_tss = 0.0
    best_train_tss = 0.0
    best_pr_auc = 0.0
    best_epoch = 0
    epoch = 0

    # gradient check
    # input = X_train_data_tensor[:10].double().requires_grad_(True).to(device)
    # test = gradcheck(model.double(), input, eps=1e-6, atol=1e-4)
    # print(test)

    # wandb artifact
    # artifact.add_file(filepath + 'normalized_training.csv',
    #                   name='normalized_training')
    # artifact.add_file(filepath + 'normalized_validation.csv',
    #                   name='normalized_validation')
    # artifact.add_file(filepath + 'normalized_testing.csv',
    #                   name='normalized_testing')
    # run.log_artifact(artifact)

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
                                  criterion, cfg, scheduler=scheduler)[5]
                stopping_metric, best_val_tss, best_pr_auc, best_epoch = validate(
                    model, device, valid_loader, criterion, epoch, best_val_tss,
                    best_pr_auc, best_epoch, cfg)[0:4]
                if best_epoch == epoch:
                    best_train_tss = train_tss

                test_tss = test(model, device, test_loader, criterion, epoch)[
                    5]

                if early_stop.step(stopping_metric) and cfg.early_stop:
                    print('[INFO] Early Stopping')
                    break

                epoch += 1

        wandb.log(
            {"Best_Validation_TSS": best_val_tss, "Best_Validation_epoch":
                best_epoch, "Best_Train_TSS": best_train_tss,
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
                                         run_path=cfg.model_name)
            model.load_state_dict(torch.load(weights_file.name))

        test_tss = test(model, device, test_loader, criterion, epoch)[5]
        wandb.log({"Test_TSS": test_tss})


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
        combined_inputs = np.concatenate([inputs, X_valid_data],
                                         axis=0).astype(np.float32)
        combined_labels = np.concatenate([labels, y_valid_tr], axis=0)
        ds = Dataset(combined_inputs, combined_labels)
        combined_labels = np.array([labels for _, labels in iter(ds)])

        # Metrics + Callbacks
        valid_tss_cb = EpochScoring(
            scoring=make_scorer(skorch_utils.get_tss, needs_proba=False),
            lower_is_better=False, name='valid_tss', use_caching=True)
        valid_bss_cb = EpochScoring(
            scoring=skorch_utils.get_bss,
            lower_is_better=False, name='valid_bss', use_caching=False)
        valid_pr_auc_cb = EpochScoring(
            scoring=make_scorer(skorch_utils.get_pr_auc, needs_proba=False),
            lower_is_better=False, name='valid_pr_auc', use_caching=True)
        train_bss_cb = EpochScoring(scoring=skorch_utils.get_bss,
            lower_is_better=False, name='train_bss', use_caching=False,
                                    on_train=True)
        train_tss_cb = EpochScoring(scoring=make_scorer(skorch_utils.get_tss,
                                                     needs_proba=False),
                                 lower_is_better=False, name='train_tss',
                                 use_caching=True, on_train=True)
        valid_hss_cb = EpochScoring(scoring=make_scorer(skorch_utils.get_hss,
                                                     needs_proba=False),
                                 lower_is_better=False, name='valid_hss',
                                 use_caching=True)
        train_bacc_cb = EpochScoring(
            scoring=make_scorer(balanced_accuracy_score, **{'adjusted': True},
                                needs_proba=False), lower_is_better=False,
            name='train_bacc', use_caching=True, on_train=True)


        if cfg.early_stop:
            earlystop = EarlyStopping(monitor='train_loss',
                                      lower_is_better=True,
                                      patience=cfg.patience)
        else:
            earlystop = None

        savename = ''
        if cfg.model_type == 'MLP':
            savename = os.path.join(wandb.run.dir,
                                    '{}_{}_{}_{}_{}_{}'.format(cfg.model_type,
                                                               cfg.layers,
                                                               cfg.hidden_units,
                                                               cfg.batch_size,
                                                               cfg.learning_rate,
                                                               cfg.seed))
        elif (cfg.model_type == 'TCN') or (cfg.model_type == 'CNN'):
            savename = os.path.join(wandb.run.dir,
                                    '{}_{}_{}_{}_{}_{}'.format(cfg.model_type,
                                                               cfg.levels,
                                                               cfg.nhid,
                                                               cfg.batch_size,
                                                               cfg.learning_rate,
                                                               cfg.seed))

        checkpoint = Checkpoint(monitor='valid_tss_best',
                                dirname=savename)
        if cfg.lr_scheduler:
            # lrscheduler = LRScheduler(policy='TorchCyclicLR',
            #                           monitor='valid_tss',
            #                           base_lr=cfg.learning_rate,
            #                           max_lr=cfg.max_lr,
            #                           step_size_up=int(4 * (
            #                                   len(X_train_data) /
            #                                   cfg.batch_size)),
            #                           cycle_momentum=True if
            #                           cfg.optim == 'SGD' else False,
            #                           step_every='batch')
            lrscheduler = LRScheduler(policy=lr_scheduler.OneCycleLR,
                                      monitor='valid_tss',
                                      max_lr=cfg.max_lr,
                                      # base_lr=cfg.max_lr/10,
                                      # step_size_up = 150,
                                      steps_per_epoch=len(train_loader),
                                      epochs=cfg.epochs,
                                      cycle_momentum=True if
                                      cfg.optim == 'SGD' else False,
                                      step_every='batch',
                                      # div_factor=25,
                                      # total_steps=5000
                                      )
            # plot lr over iterations
            lrs = lrscheduler.simulate(cfg.epochs*len(train_loader),
                                       cfg.max_lr)
            plt.plot(lrs)
            plt.ylabel('Learning rate')
            plt.xlabel('Iterations')
            plt.show()
            # for i in range(len(lrs)):
            #     wandb.log({'Learning Rate': lrs[i], 'Iterations': i})
        else:
            lrscheduler = None

        logger = skorch_utils.LoggingCallback(test_inputs, test_labels)

        load_state = LoadInitState(checkpoint)
        reload_at_end = skorch_utils.LoadBestCP(checkpoint)

        # noinspection PyArgumentList
        net = NeuralNetClassifier(model, max_epochs=cfg.epochs,
                                  batch_size=cfg.batch_size,
                                  criterion=nn.CrossEntropyLoss,
                                  criterion__weight=torch.FloatTensor(
                                      class_weights).to(device),
                                  optimizer=torch.optim.SGD,
                                  optimizer__lr=cfg.learning_rate,
                                  optimizer__weight_decay=cfg.weight_decay,
                                  device=device,
                                  train_split=predefined_split(valid_ds),
                                  # train_split=skorch.dataset.CVSplit(cv=10),
                                  callbacks=[train_tss_cb,
                                             train_bss_cb, valid_tss_cb,
                                             valid_hss_cb, valid_bss_cb,
                                             earlystop, checkpoint,
                                             # load_state,
                                             # reload_at_end,
                                             logger,
                                             lrscheduler],
                                  iterator_train__shuffle=True if cfg.shuffle else False,
                                  warm_start=False)

        # set optimizer dynamically
        if cfg.optim == 'SGD':
            net.set_params(optimizer=torch.optim.SGD)
            net.set_params(optimizer__momentum=cfg.momentum)
            net.set_params(optimizer__nesterov=True)
        else:
            net.set_params(optimizer=torch.optim.Adam)

        net.initialize()
        init_savename = os.path.join(wandb.run.dir,
                                     'init_{}_{}'.format(cfg.model_type,
                                                         cfg.seed))
        net.save_params(f_params=init_savename)

        if not (cfg.cross_validation) and not (cfg.nested_cv) and (cfg.training):
            net.fit(inputs, labels)

        # '''
        # Cross Validation
        # '''
        elif cfg.cross_validation:
            kf = sklearn.model_selection.KFold(n_splits=cfg.n_splits, shuffle=False)
            skf = sklearn.model_selection.StratifiedKFold(cfg.n_splits, shuffle=False)
            tscv = sklearn.model_selection.TimeSeriesSplit(n_splits=cfg.n_splits)
            rkf = sklearn.model_selection.RepeatedKFold(n_splits=cfg.n_splits,
                                                        n_repeats=2)
            scores = []
            visualize_CV.visualize_cv(sklearn.model_selection.KFold,
                                      combined_inputs, combined_labels, cfg)
            for train_index, val_index in kf.split(combined_inputs,
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
            wandb.log({"CV_Score": scores})

        elif cfg.nested_cv:
            gcv = do_nested_cv()


        '''
        Test Results
        '''
        # train on train and val set
        if cfg.cross_validation:
            y_test = net.predict(test_inputs)
            tss_test_score = skorch_utils.get_tss(test_labels, y_test)
            wandb.log({'Test_TSS': tss_test_score})
            print("Test TSS:" + str(tss_test_score))

        elif cfg.nested_cv:
            tss_test_score = balanced_accuracy_score(y_true=test_labels,
                                                     y_pred=gcv.predict(
                                                         test_inputs),
                                                     adjusted=True)
            wandb.log({'Test_TSS': tss_test_score})
            print('Test TSS: %.4f' % (tss_test_score))

        else:
            if cfg.checkpoint:
                net.initialize()
                net.load_params(checkpoint=checkpoint)  # Select best TSS epoch
            else:
                pass
            y_test = net.predict(test_inputs)
            tss_test_score = skorch_utils.get_tss(test_labels, y_test)
            hss_test_score = skorch_utils.get_hss(test_labels, y_test)
            wandb.log({'Test_TSS': tss_test_score})
            wandb.log({'Test_HSS': hss_test_score})
            print("Test TSS:" + str(tss_test_score))
            print("Test HSS:" + str(hss_test_score))
    # Save model to W&B
    torch.save(model.state_dict(), os.path.join(wandb.run.dir, 'model.pt'))


    '''
    Model interpretation
    '''
    if cfg.evaluation:
        # all_ones = np.ones(inputs.shape[0])
        # all_zeros = np.zeros(inputs.shape[0])
        # all_fill = np.full(inputs.shape[0], 0.01)
        device = 'cpu'
        model = model.to(device)
        # model.device = device

        # Train
        # yprob = infer_model(model, device, train_loader)
        yprob = metric.get_proba(model(X_train_data_tensor.to(device)))[:,1]
        # yprob = pd.read_csv('./saved/results/liu/train.csv').to_numpy()

        pdf.plot_eval_graphs(yprob, y_train_tr_tensor.numpy(), 'Train')
        metric.plot_precision_recall(model, yprob, y_train_tr_tensor, 'Train')
        metric.plot_confusion_matrix(yprob, y_train_tr_tensor, 'Train')
        roc_auc = metric.get_roc(model, yprob, y_train_tr_tensor, device, 'Train')
        th = metric.get_metrics_threshold(yprob, y_train_tr_tensor)[11]
        pdf.plot_density_estimation(model, yprob, y_train_tr_tensor, 'Train')

        cm = sklearn.metrics.confusion_matrix(y_train_tr_tensor,
                                              metric.to_labels(yprob, 0.5))
        tss_train = metric.calculate_metrics(cm, 2)[4]


        # Validation
        # yprob = infer_model(model, device, valid_loader)
        yprob = metric.get_proba(model(X_valid_data_tensor.to(device)))[:,1]
        # yprob = pd.read_csv('./saved/results/liu/val.csv').to_numpy()

        pdf.plot_eval_graphs(yprob, y_valid_tr_tensor.numpy(), 'Validation')
        metric.plot_precision_recall(model, yprob, y_valid_tr_tensor, 'Validation')
        metric.plot_confusion_matrix(yprob, y_valid_tr_tensor, 'Validation')
        roc_auc = metric.get_roc(model, yprob, y_valid_tr_tensor, device, 'Validation')
        th_norm = pdf.plot_density_estimation(model, yprob, y_valid_tr_tensor,'Validation')
        th = metric.get_metrics_threshold(yprob, y_valid_tr_tensor)[11]

        cm = sklearn.metrics.confusion_matrix(y_valid_tr_tensor,
                                              metric.to_labels(yprob, 0.5))
        tss_val = metric.calculate_metrics(cm, 2)[4]


        # Test
        # yprob = infer_model(model, device, test_loader)
        yprob = metric.get_proba(model(X_test_data_tensor.to(device)))[:,1]
        # yprob = pd.read_csv('./saved/results/liu/test.csv').to_numpy()

        metric.plot_precision_recall(model, yprob, y_test_tr_tensor, 'Test')
        pdf.plot_eval_graphs(yprob, y_test_tr_tensor.numpy(), 'Test')
        cm = sklearn.metrics.confusion_matrix(y_test_tr_tensor,
                                              metric.to_labels(yprob, th))
        tss_th = metric.calculate_metrics(cm, 2)[4]
        metric.plot_confusion_matrix(yprob, y_test_tr_tensor, 'Test')
        tss = metric.get_metrics_threshold(yprob, y_test_tr_tensor)[8]
        roc_auc = metric.get_roc(model, yprob, y_test_tr_tensor, device, 'Test')
        th_norm_test = pdf.plot_density_estimation(model, yprob, y_test_tr_tensor, 'Test')

        print(
            'Test TSS from validation threshold ({:0.3f}): {:0.3f}'.format(th, tss_th))
        wandb.log({'Test_TSS_Th': tss_th})



        '''
        Attribution methods
        '''
    if cfg.interpret:
        df = pd.read_csv(filepath + 'normalized_testing.csv')
        if not os.path.exists(filepath + 'Case_Study/'):
            os.makedirs(filepath + 'Case_Study/')
        case = df[df['NOAA'] == 12673]
        case2 = df[df['NOAA'] == 12252]
        # case = case.iloc[len(case)-len(case2):,:]
        case.to_csv(filepath + 'Case_Study/AR12673.csv', index=False)
        case2.to_csv(filepath + 'Case_Study/AR12252.csv', index=False)

        # get samples to interpret
        input_df, _ = data_loader.load_data(
            datafile=filepath+'Case_Study/AR12673.csv',
            flare_label=cfg.flare_label, series_len=cfg.seq_len,
            start_feature=start_feature, n_features=cfg.n_features,
            mask_value=mask_value, feature_list=feature_list)

        # todo pass median or weighted k-medians values as background?
        backgroud_df, _ = data_loader.load_data(
            datafile=filepath+'Case_Study/AR12252.csv',
            flare_label=cfg.flare_label, series_len=cfg.seq_len,
            start_feature=start_feature, n_features=cfg.n_features,
            mask_value=mask_value, feature_list=feature_list)

        if cfg.model_type == 'MLP':
            input_df = np.reshape(input_df, (len(input_df), cfg.n_features))
            backgroud_df = np.reshape(backgroud_df, (len(backgroud_df), cfg.n_features))
        else:
            backgroud_df = torch.tensor(backgroud_df).float().permute(0, 2, 1)
            input_df = torch.tensor(input_df).float().permute(0, 2, 1)

        # interpret using captum
        attrs_list = interpreter.interpret_model(model, device, input_df,
                                                 backgroud_df)

        attr_name_list = ["Saliency", "Integrated Gradients", "DeepLIFT",
                          "Input x Gradient", "Guided Backprop", "Ablation",
                          "Shapley Value Sampling"]

        # interpreter.plot_all_attr(attrs_list, feature_list, attr_name_list)
        interpreter.plot_attr_vs_time(attrs_list, feature_list, attr_name_list)
        interpreter.log_attrs(attrs_list, feature_list, attr_name_list, cfg)

        # visualize interpretation
        # interpreter.visualize_importance(np.array(
        #     feature_names[start_feature:start_feature + cfg.n_features]),
        #     attr_sal, cfg.n_features, title="Saliency")
        # interpreter.visualize_importance(np.array(
        #     feature_names[start_feature:start_feature + cfg.n_features]),
        #     attr_ig, cfg.n_features, title="Integrated Gradients")
        # interpreter.visualize_importance(np.array(
        #     feature_names[start_feature:start_feature + cfg.n_features]),
        #     attr_dl, cfg.n_features, title="DeepLIFT")
        # interpreter.visualize_importance(np.array(
        #     feature_names[start_feature:start_feature + cfg.n_features]),
        #     attr_ixg, cfg.n_features, title="Input x Gradient")
        # interpreter.visualize_importance(np.array(
        #     feature_names[start_feature:start_feature + cfg.n_features]),
        #     attr_gbp, cfg.n_features, title="Guided Backprop")
        # interpreter.visualize_importance(np.array(
        #     feature_names[start_feature:start_feature + cfg.n_features]),
        #     attr_occ, cfg.n_features, title="Occlusion")
        # interpreter.visualize_importance(np.array(
        #     feature_names[start_feature:start_feature + cfg.n_features]),
        #     attr_abl, cfg.n_features, title="Ablation")
        # interpreter.visualize_importance(np.array(
        #     feature_names[start_feature:start_feature + cfg.n_features]),
        #     attr_shap, cfg.n_features, title="Shapley Value Sampling")

        '''SHAP'''
        plt.close('all')
        interpreter.get_shap(model, input_df, backgroud_df, device, cfg,
                             feature_list, start_feature)

    print('Finished')

