'''
Ecclesiastes 5:12 New King James Version (NKJV)
12 The sleep of a laboring man is sweet,
Whether he eats little or much;
But the abundance of the rich will not permit him to sleep.
'''
import pandas as pd
from matplotlib import pyplot
from sklearn.utils import class_weight
from sklearn.metrics import make_scorer
import sklearn.metrics

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
from skorch_tools import skorch_utils

from data_loader import CustomDataset
from data_loader import data_loader
from utils import early_stopping
from model.tcn import TemporalConvNet
from model import metric
from interpret import interpreter

import wandb

'''
TCN with n residual blocks will have a receptive field of
1 + 2*(kernel_size-1)*(2^n-1)
'''


def preprocess_customdataset(x_val, y_val):
    # change format to tensors and create data set
    # x_tensor = torch.tensor(x_val).type(torch.FloatTensor)
    # y_tensor = torch.tensor(y_val).type(torch.LongTensor)
    # y_tensor = torch.nn.functional.one_hot(torch.LongTensor(y_val), 2)

    datasets = {}
    # datasets = CustomDataset.CustomDataset(x_tensor, y_tensor)
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
        y1 = self.tcn(x)  # input should have dimension (N, C, L)
        return self.linear(y1[:, :, -1])


def train(model, device, train_loader, optimizer, epoch, criterion):
    start = time.time()
    model.train()
    confusion_matrix = torch.zeros(nclass, nclass)
    loss_epoch = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss_epoch += criterion(output, target).item()
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(output.data, 1)

        for t, p in zip(target.view(-1), predicted.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1

        if batch_idx % args.log_interval == 0:
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))

    loss_epoch /= len(train_loader.dataset)
    print("Training Scores:")
    recall, precision, accuracy, bacc, tss, hss, tp, fn, fp, \
    tn = metric.calculate_metrics(
        confusion_matrix, nclass)

    end = time.time()
    print(end - start)

    wandb.log({"Training_Accuracy": accuracy[0], "Training_TSS": tss[0],
               "Training_HSS": hss[0], "Training_BACC": bacc[0],
               "Training_Precision": precision[1],
               "Training_Recall": recall[1], "Training_Loss": loss_epoch},
              step=epoch)


def validate(model, device, valid_loader, criterion, epoch, best_tss,
             best_epoch):
    model.eval()
    valid_loss = 0
    correct = 0
    confusion_matrix = torch.zeros(nclass, nclass)

    example_images = []
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
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({'
          ':.0f}%)\n'.format(valid_loss, correct, len(valid_loader.dataset),
                             100. * correct / len(valid_loader.dataset)))

    # # validation conf matrix
    # print(confusion_matrix)
    # # per class accuracy
    # print(confusion_matrix.diag() / confusion_matrix.sum(1))

    print("Validation Scores:")
    recall, precision, accuracy, bacc, tss, hss, tp, fn, fp, \
    tn = metric.calculate_metrics(
        confusion_matrix, nclass)

    wandb.log({"Validation_Accuracy": accuracy[0], "Validation_TSS": tss[0],
               "Validation_HSS": hss[0], "Validation_BACC": bacc[0],
               "Validation_Precision": precision[1],
               "Validation_Recall": recall[1], "Validation_Loss": valid_loss},
              step=epoch)

    # checkpoint on best tss
    if tss[0] >= best_tss:
        best_tss = tss[0]
        best_epoch = epoch
        torch.save(model.state_dict(), os.path.join(wandb.run.dir, 'model.pt'))
        print('[INFO] Checkpointing...')

    return tss[0], best_tss, best_epoch


def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    confusion_matrix = torch.zeros(nclass, nclass)

    example_images = []
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
    print(
        '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

    # # testation conf matrix
    # print(confusion_matrix)
    # # per class accuracy
    # print(confusion_matrix.diag() / confusion_matrix.sum(1))

    print("Test Scores:")
    recall, precision, accuracy, bacc, tss, hss, tp, fn, fp, \
    tn = metric.calculate_metrics(
        confusion_matrix, nclass)

    wandb.log(
        {"Test_Accuracy": accuracy[0], "Test_TSS": tss[0], "Test_HSS": hss[0],
         "Test_BACC": bacc[0], "Test_Precision": precision[1],
         "Test_Recall": recall[1], "Test_Loss": test_loss})


def infer_model(model, device, data_loader):
    model.eval()
    output_arr = []
    with torch.no_grad():
        # output = model(data_loader.dataset.data.to(device))
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            output_arr.append(output.cpu().detach().numpy())

    return np.vstack(output_arr)


if __name__ == '__main__':
    wandb.init(project="liu_pytorch_tcn", notes='TCN')


    # parse hyperparameters
    parser = argparse.ArgumentParser(description='Deep Flare Prediction')
    parser.add_argument('--epochs', type=int, default=1,
                        help='upper epoch limit (default: 100)')
    parser.add_argument('--flare_label', default="M5",
                        help='Types of flare class (default: M-Class')
    parser.add_argument('--batch_size', type=int, default=2048, metavar='N',
                        help='input batch size for training (default: 256)')
    parser.add_argument('--learning_rate', type=float, default=2e-4,
                        help='initial learning rate (default: 1e-3)')
    parser.add_argument('--layer_dim', type=int, default=1, metavar='N',
                        help='how many hidden layers (default: 5)')

    parser.add_argument('--levels', type=int, default=1,
                        help='# of levels (default: 4)')
    parser.add_argument('--ksize', type=int, default=2,
                        help='kernel size (default: 5)')
    parser.add_argument('--nhid', type=int, default=20,
                        help='number of hidden units per layer (default: 20)')
    parser.add_argument('--n_features', type=int, default=20,
                        help='number of features (default: 20)')

    # parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
    #                     help='SGD momentum (default: 0.5)')

    parser.add_argument('--dropout', type=float, default=0.7,
                        help='dropout applied to layers (default: 0.7)')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                        metavar='LR', help='L2 regularizing (default: 0.0001)')
    parser.add_argument('--rnn_module', default="TCN",
                        help='Types of rnn (default: LSTM')

    # parser.add_argument('--clip', type=float, default=0.2,
    #                     help='gradient clip, -1 means no clip (default:
    #                     0.2)')
    parser.add_argument('--optim', type=str, default='Adam',
                        help='optimizer to use (default: Adam)')
    parser.add_argument('--seed', type=int, default=5,
                        help='random seed (default: 1111)')
    parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                        help='report interval (default: 100')
    parser.add_argument('--cuda', action='store_false', default=True,
                        help='enables CUDA training')
    parser.add_argument('--early_stop', action='store_true', default=True,
                        help='Stops training if overfitting')
    parser.add_argument('--restore', action='store_true', default=False,
                        help='restores model')
    parser.add_argument('--training', action='store_true', default=True,
                        help='trains and test model, if false only tests')
    args = parser.parse_args()
    wandb.config.update(args)

    # initialize parameters
    filepath = './Data/Liu/' + args.flare_label + '/'
    num_of_fold = 10
    n_features = 0
    if args.flare_label == 'M5':
        n_features = args.n_features  # 20 original
    elif args.flare_label == 'M':
        n_features = 22
    elif args.flare_label == 'C':
        n_features = 14
    feature_names = data_loader.get_feature_names(
        filepath + 'normalized_training.csv')

    # initialize parameters
    start_feature = 5
    mask_value = 0
    nclass = 2

    # GPU check
    use_cuda = args.cuda and torch.cuda.is_available()
    if args.cuda == True and torch.cuda.is_available():
        print("Cuda enabled and available")
    elif args.cuda == True and torch.cuda.is_available() == False:
        print("Cuda enabled not not available, CPU used.")
    elif args.cuda == False:
        print("Cuda disabled")
    device = torch.device("cuda" if use_cuda else "cpu")

    # set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    np.random.seed(args.seed)

    # setup dataloaders
    X_train_data, y_train_data = data_loader.load_data(
        datafile=filepath + 'normalized_training.csv',
        flare_label=args.flare_label, series_len=args.layer_dim,
        start_feature=start_feature, n_features=n_features,
        mask_value=mask_value)

    X_valid_data, y_valid_data = data_loader.load_data(
        datafile=filepath + 'normalized_validation.csv',
        flare_label=args.flare_label, series_len=args.layer_dim,
        start_feature=start_feature, n_features=n_features,
        mask_value=mask_value)

    X_test_data, y_test_data = data_loader.load_data(
        datafile=filepath + 'normalized_testing.csv',
        flare_label=args.flare_label, series_len=args.layer_dim,
        start_feature=start_feature, n_features=n_features,
        mask_value=mask_value)

    y_train_tr = data_loader.label_transform(y_train_data)
    y_valid_tr = data_loader.label_transform(y_valid_data)
    y_test_tr = data_loader.label_transform(y_test_data)

    X_train_data_tensor = torch.tensor(X_train_data).float()
    X_train_data_tensor = X_train_data_tensor.view(len(X_train_data_tensor),
                                                   n_features, args.layer_dim)
    y_train_tr_tensor = torch.tensor(y_train_tr).long()

    X_valid_data_tensor = torch.tensor(X_valid_data).float()
    X_valid_data_tensor = X_valid_data_tensor.view(len(X_valid_data_tensor),
                                                   n_features, args.layer_dim)
    y_valid_tr_tensor = torch.tensor(y_valid_tr).long()

    X_test_data_tensor = torch.tensor(X_test_data).float()
    X_test_data_tensor = X_test_data_tensor.view(len(X_test_data_tensor),
                                                 n_features, args.layer_dim)
    y_test_tr_tensor = torch.tensor(y_test_tr).long()

    # ready custom dataset
    datasets = {}
    datasets['train'] = preprocess_customdataset(X_train_data_tensor,
                                                 y_train_tr_tensor)
    datasets['valid'] = preprocess_customdataset(X_valid_data_tensor,
                                                 y_valid_tr_tensor)
    datasets['test'] = preprocess_customdataset(X_test_data_tensor,
                                                y_test_tr_tensor)

    train_loader = torch.utils.data.DataLoader(datasets['train'],
                                               args.batch_size, shuffle=False,
                                               drop_last=False)
    valid_loader = torch.utils.data.DataLoader(datasets['valid'],
                                               args.batch_size, shuffle=False,
                                               drop_last=False)
    test_loader = torch.utils.data.DataLoader(datasets['test'],
                                              args.batch_size, shuffle=False,
                                              drop_last=False)

    # make model
    channel_sizes = [args.nhid] * args.levels
    kernel_size = args.ksize
    dropout = args.dropout

    model = TCN(n_features, nclass, channel_sizes, kernel_size=kernel_size,
                dropout=dropout).to(device)
    wandb.watch(model, log='all')

    # optimizers
    class_weights = class_weight.compute_class_weight('balanced',
                                                      np.unique(y_train_data),
                                                      y_train_data)

    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(
        device))  # weighted cross entropy
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate,
                                 weight_decay=args.weight_decay, amsgrad=False)

    # print model parameters
    print("Receptive Field: " + str(1+2*(args.ksize-1)*(2 ^ args.levels - 1)))
    # print(len(list(model.parameters())))
    for i in range(len(list(model.parameters()))):
        print(list(model.parameters())[i].size())

    # early stopping check
    early_stop = early_stopping.EarlyStopping(mode='max', patience=40)
    best_tss = 0.0
    best_epoch = 0

    print("Training Network...")
    if args.training:
        epoch = 0
        while epoch < args.epochs:
            train(model, device, train_loader, optimizer, epoch, criterion)
            tss, best_tss, best_epoch = validate(model, device, valid_loader,
                                                 criterion, epoch, best_tss,
                                                 best_epoch)

            if early_stop.step(tss) and args.early_stop:
                print('[INFO] Early Stopping')
                break

            # Continue training if recently improved
            if epoch == args.epochs-1 and early_stop.num_bad_epochs < 2:
                args.epochs += 5
                print("[INFO] not finished training...")
            epoch += 1

    wandb.log(
        {"Best_Validation_TSS": best_tss, "Best_Validation_epoch": best_epoch})

    # reload best tss checkpoint and test
    print("[INFO] Loading model at epoch:" + str(best_epoch))
    try:
        model.load_state_dict(
            torch.load(os.path.join(wandb.run.dir, 'model.pt')))
    except:
        print('No model loaded...')

    test(model, device, test_loader, criterion)

    '''
    PR Curves
    '''
    # get predicted output probabilities => numpy array
    yhat = infer_model(model, device, train_loader)

    # PR curves on test set
    precision, recall, f1, pr_auc = metric.get_precision_recall(model,
                                                                yhat,
                                                                y_train_tr_tensor,
                                                                device,
                                                                'Train')

    # get predicted output probabilities => numpy array
    yhat = infer_model(model, device, valid_loader)

    # PR curves on test set
    precision, recall, f1, pr_auc = metric.get_precision_recall(model,
                                                                yhat,
                                                                y_valid_tr_tensor,
                                                                device,
                                                                'Validation')
    # get predicted output probabilities => numpy array
    yhat = infer_model(model, device, test_loader)

    # PR curves on test set
    precision, recall, f1, pr_auc = metric.get_precision_recall(model,
                                                                yhat,
                                                                y_test_tr_tensor,
                                                                device,
                                                                'Test')

    roc_auc = metric.get_roc(model, yhat, y_test_tr_tensor, device)

    '''
    Model interpretation
    '''
    # todo interpret on test set?

    test_loader_interpret = torch.utils.data.DataLoader(datasets['test'],
                                                        int(args.batch_size/6),
                                                        shuffle=False,
                                                        drop_last=False)

    attr_ig, attr_sal, attr_ig_avg, attr_sal_avg = interpreter.interpret_model(
        model, device, test_loader_interpret, n_features, args)

    interpreter.visualize_importance(
        np.array(feature_names[start_feature:start_feature + n_features]),
        np.mean(attr_ig_avg, axis=0), np.std(attr_ig_avg, axis=0), n_features,
        title="Integrated Gradient Features")

    interpreter.visualize_importance(
        np.array(feature_names[start_feature:start_feature + n_features]),
        np.mean(attr_sal_avg, axis=0), np.std(attr_sal_avg, axis=0),
        n_features, title="Saliency Features")

    '''
        Skorch training
    '''
    # inputs = torch.tensor(X_train_data).float()
    # inputs = inputs.view(len(inputs), n_features, args.layer_dim)
    # labels = torch.tensor(y_train_tr).long()
    # X_valid_data = torch.tensor(X_valid_data).float()
    # X_valid_data = X_valid_data.view(len(X_valid_data), n_features,
    #                                  args.layer_dim)
    # y_valid_tr = torch.tensor(y_valid_tr).long()
    #
    # inputs = inputs.numpy()
    # labels = labels.numpy()
    #
    # X_valid_data = X_valid_data.numpy()
    # y_valid_tr = y_valid_tr.numpy()
    # valid_ds = Dataset(X_valid_data, y_valid_tr)
    #
    # # Metrics + Callbacks
    # valid_tss = EpochScoring(scoring=make_scorer(skorch_utils.get_tss),
    #                          lower_is_better=False, name='valid_tss',
    #                          use_caching=True)
    # valid_hss = EpochScoring(scoring=make_scorer(skorch_utils.get_hss),
    #                          lower_is_better=False, name='valid_hss',
    #                          use_caching=True)
    #
    # earlystop = EarlyStopping(monitor='valid_tss', lower_is_better=False,
    #                           patience=30)
    # checkpoint = Checkpoint(monitor='valid_tss_best',
    #                         dirname='./saved/models/exp1')
    #
    # net = NeuralNetClassifier(model, max_epochs=args.epochs,
    #                           batch_size=args.batch_size,
    #                           criterion=nn.CrossEntropyLoss,
    #                           criterion__weight=torch.FloatTensor(
    #                               class_weights).to(device),
    #                           optimizer=torch.optim.Adam,
    #                           optimizer__lr=args.learning_rate,
    #                           optimizer__weight_decay=args.weight_decay,
    #                           optimizer__amsgrad=True,
    #                           device=device,
    #                           # train_split=None, #die breek die logs
    #                           train_split=predefined_split(valid_ds),
    #                           callbacks=[valid_tss, valid_hss, earlystop,
    #                                      checkpoint,
    #                                      skorch_utils.LoggingCallback],
    #                           # iterator_train__shuffle=True, # batches
    #                           # shuffle=False
    #                           # warm_start=False
    #                           )
    #
    # net.fit(inputs, labels)
    #
    # '''
    # Test Results
    # '''
    # net.initialize()
    # net.load_params(checkpoint=checkpoint)  # Select best TSS epoch
    #
    # inputs = torch.tensor(X_test_data).float()
    # inputs = inputs.view(len(inputs), n_features, args.layer_dim)
    # labels = torch.tensor(y_test_tr).long()
    #
    # inputs = inputs.numpy()
    # labels = labels.numpy()
    #
    # y_test = net.predict(inputs)
    # tss_test_score = skorch_utils.get_tss(labels, y_test)
    # wandb.log({'Test_TSS': tss_test_score})
    # print("Test TSS:" + str(tss_test_score))

    # Save model to W&B
    torch.save(model.state_dict(), os.path.join(wandb.run.dir, 'model.pt'))
    print('Finished')
