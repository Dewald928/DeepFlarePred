'''
Ecclesiastes 5:12 New King James Version (NKJV)
12 The sleep of a laboring man is sweet,
Whether he eats little or much;
But the abundance of the rich will not permit him to sleep.
'''
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate

import sys
import os
import numpy as np
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

# import skorch
from skorch import NeuralNetClassifier
from skorch.callbacks import EpochScoring
from skorch.callbacks import *
from skorch.helper import predefined_split
from skorch.dataset import Dataset

from data_loader import CustomDataset
from utils import early_stopping
import wandb


def load_data(datafile, flare_label, series_len, start_feature, n_features, mask_value):
    df = pd.read_csv(datafile)
    df_values = df.values
    X = []
    y = []
    tmp = []
    for k in range(start_feature, start_feature + n_features):
        tmp.append(mask_value)
    for idx in range(0, len(df_values)):
        each_series_data = []
        row = df_values[idx]
        if flare_label == 'M5' and row[1][0] == 'M' and float(row[1][1:]) >= 5.0:
            label = 'X'
        else:
            label = row[1][0]
        if flare_label == 'M' and label == 'X':
            label = 'M'
        if flare_label == 'C' and (label == 'X' or label == 'M'):
            label = 'C'
        if flare_label == 'B' and (label == 'X' or label == 'M' or label == 'C'):
            label = 'B'
        if flare_label == 'M5' and (label == 'M' or label == 'C' or label == 'B'):
            label = 'N'
        if flare_label == 'M' and (label == 'B' or label == 'C'):
            label = 'N'
        if flare_label == 'C' and label == 'B':
            label = 'N'
        has_zero_record = False
        # if at least one of the 25 physical feature values is missing, then discard it.
        if flare_label == 'C':
            if float(row[5]) == 0.0:
                has_zero_record = True
            if float(row[7]) == 0.0:
                has_zero_record = True
            for k in range(9, 13):
                if float(row[k]) == 0.0:
                    has_zero_record = True
                    break
            for k in range(14, 16):
                if float(row[k]) == 0.0:
                    has_zero_record = True
                    break
            for k in range(18, 21):
                if float(row[k]) == 0.0:
                    has_zero_record = True
                    break
            if float(row[22]) == 0.0:
                has_zero_record = True
            for k in range(24, 33):
                if float(row[k]) == 0.0:
                    has_zero_record = True
                    break
            for k in range(38, 42):
                if float(row[k]) == 0.0:
                    has_zero_record = True
                    break
        elif flare_label == 'M':
            for k in range(5, 10):
                if float(row[k]) == 0.0:
                    has_zero_record = True
                    break
            for k in range(13, 16):
                if float(row[k]) == 0.0:
                    has_zero_record = True
                    break
            if float(row[19]) == 0.0:
                has_zero_record = True
            if float(row[21]) == 0.0:
                has_zero_record = True
            for k in range(23, 30):
                if float(row[k]) == 0.0:
                    has_zero_record = True
                    break
            for k in range(31, 33):
                if float(row[k]) == 0.0:
                    has_zero_record = True
                    break
            for k in range(34, 37):
                if float(row[k]) == 0.0:
                    has_zero_record = True
                    break
            for k in range(39, 41):
                if float(row[k]) == 0.0:
                    has_zero_record = True
                    break
            if float(row[42]) == 0.0:
                has_zero_record = True
        elif flare_label == 'M5':
            for k in range(5, 12):
                if float(row[k]) == 0.0:
                    has_zero_record = True
                    break
            for k in range(19, 21):
                if float(row[k]) == 0.0:
                    has_zero_record = True
                    break
            for k in range(22, 31):
                if float(row[k]) == 0.0:
                    has_zero_record = True
                    break
            for k in range(32, 37):
                if float(row[k]) == 0.0:
                    has_zero_record = True
                    break
            for k in range(40, 42):
                if float(row[k]) == 0.0:
                    has_zero_record = True
                    break

        if has_zero_record is False:
            cur_noaa_num = int(row[3])
            each_series_data.append(row[start_feature:start_feature + n_features].tolist())
            itr_idx = idx - 1
            while itr_idx >= 0 and len(each_series_data) < series_len:
                prev_row = df_values[itr_idx]
                prev_noaa_num = int(prev_row[3])
                if prev_noaa_num != cur_noaa_num:
                    break
                has_zero_record_tmp = False
                if flare_label == 'C':
                    if float(row[5]) == 0.0:
                        has_zero_record_tmp = True
                    if float(row[7]) == 0.0:
                        has_zero_record_tmp = True
                    for k in range(9, 13):
                        if float(row[k]) == 0.0:
                            has_zero_record_tmp = True
                            break
                    for k in range(14, 16):
                        if float(row[k]) == 0.0:
                            has_zero_record_tmp = True
                            break
                    for k in range(18, 21):
                        if float(row[k]) == 0.0:
                            has_zero_record_tmp = True
                            break
                    if float(row[22]) == 0.0:
                        has_zero_record_tmp = True
                    for k in range(24, 33):
                        if float(row[k]) == 0.0:
                            has_zero_record_tmp = True
                            break
                    for k in range(38, 42):
                        if float(row[k]) == 0.0:
                            has_zero_record_tmp = True
                            break
                elif flare_label == 'M':
                    for k in range(5, 10):
                        if float(row[k]) == 0.0:
                            has_zero_record_tmp = True
                            break
                    for k in range(13, 16):
                        if float(row[k]) == 0.0:
                            has_zero_record_tmp = True
                            break
                    if float(row[19]) == 0.0:
                        has_zero_record_tmp = True
                    if float(row[21]) == 0.0:
                        has_zero_record_tmp = True
                    for k in range(23, 30):
                        if float(row[k]) == 0.0:
                            has_zero_record_tmp = True
                            break
                    for k in range(31, 33):
                        if float(row[k]) == 0.0:
                            has_zero_record_tmp = True
                            break
                    for k in range(34, 37):
                        if float(row[k]) == 0.0:
                            has_zero_record_tmp = True
                            break
                    for k in range(39, 41):
                        if float(row[k]) == 0.0:
                            has_zero_record_tmp = True
                            break
                    if float(row[42]) == 0.0:
                        has_zero_record_tmp = True
                elif flare_label == 'M5':
                    for k in range(5, 12):
                        if float(row[k]) == 0.0:
                            has_zero_record_tmp = True
                            break
                    for k in range(19, 21):
                        if float(row[k]) == 0.0:
                            has_zero_record_tmp = True
                            break
                    for k in range(22, 31):
                        if float(row[k]) == 0.0:
                            has_zero_record_tmp = True
                            break
                    for k in range(32, 37):
                        if float(row[k]) == 0.0:
                            has_zero_record_tmp = True
                            break
                    for k in range(40, 42):
                        if float(row[k]) == 0.0:
                            has_zero_record_tmp = True
                            break

                if len(each_series_data) < series_len and has_zero_record_tmp is True:
                    each_series_data.insert(0, tmp)

                if len(each_series_data) < series_len and has_zero_record_tmp is False:
                    each_series_data.insert(0, prev_row[start_feature:start_feature + n_features].tolist())
                itr_idx -= 1

            while len(each_series_data) > 0 and len(each_series_data) < series_len:
                each_series_data.insert(0, tmp)

            if len(each_series_data) > 0:
                X.append(np.array(each_series_data).reshape(series_len, n_features).tolist())
                y.append(label)
    X_arr = np.array(X)
    y_arr = np.array(y)
    print(X_arr.shape)
    return X_arr, y_arr


def label_transform(data):
    encoder = LabelEncoder()
    encoder.fit(data)
    encoded_Y = encoder.transform(data)
    # converteddata = np.eye(nclass, dtype='uint8')[encoded_Y]
    return encoded_Y


def preprocess_customdataset(x_val, y_val):
    # change format to tensors and create data set
    x_tensor = torch.tensor(x_val).type(torch.FloatTensor)
    y_tensor = torch.tensor(y_val).type(torch.LongTensor)
    # y_tensor = torch.nn.functional.one_hot(torch.LongTensor(y_val), 2)

    datasets = {}
    datasets = CustomDataset.CustomDataset(x_tensor, y_tensor)

    return datasets


def calculate_metrics(confusion_matrix):
    # determine skill scores
    print('Calculating skill scores: ')
    confusion_matrix = confusion_matrix.numpy()
    N = np.sum(confusion_matrix)

    recall = np.zeros(nclass)
    precision = np.zeros(nclass)
    accuracy = np.zeros(nclass)
    bacc = np.zeros(nclass)
    tss = np.zeros(nclass)
    hss = np.zeros(nclass)
    tp = np.zeros(nclass)
    fn = np.zeros(nclass)
    fp = np.zeros(nclass)
    tn = np.zeros(nclass)
    for p in range(nclass):
        tp[p] = confusion_matrix[p][p]
        for q in range(nclass):
            if q != p:
                fn[p] += confusion_matrix[p][q]
                fp[p] += confusion_matrix[q][p]
        tn[p] = N - tp[p] - fn[p] - fp[p]

        recall[p] = round(float(tp[p]) / float(tp[p] + fn[p] + 1e-6), 3)
        precision[p] = round(float(tp[p]) / float(tp[p] + fp[p] + 1e-6), 3)
        accuracy[p] = round(float(tp[p] + tn[p]) / float(N), 3)
        bacc[p] = round(
            0.5 * (float(tp[p]) / float(tp[p] + fn[p]) + float(tn[p]) / float(tn[p] + fp[p])), 3)
        hss[p] = round(2 * float(tp[p] * tn[p] - fp[p] * fn[p])
                       / float((tp[p] + fn[p]) * (fn[p] + tn[p])
                               + (tp[p] + fp[p]) * (fp[p] + tn[p])), 3)
        tss[p] = round(
            (float(tp[p]) / float(tp[p] + fn[p] + 1e-6) - float(fp[p]) / float(fp[p] + tn[p] + 1e-6)), 3)

    print("tss: " + str(tss))
    print("hss: " + str(hss))
    print("bacc: " + str(bacc))
    print("accuracy: " + str(accuracy))
    print("precision: " + str(precision))
    print("recall: " + str(recall))

    return recall, precision, accuracy, bacc, tss, hss, tp, fn, fp, tn


def get_tss(y_true, y_pred):
    # determine skill scores
    import sklearn.metrics
    # print('Calculating skill scores: ')
    confusion_matrix = []
    confusion_matrix = sklearn.metrics.confusion_matrix(y_true, y_pred)
    N = np.sum(confusion_matrix)

    recall = np.zeros(nclass)
    precision = np.zeros(nclass)
    accuracy = np.zeros(nclass)
    bacc = np.zeros(nclass)
    tss = np.zeros(nclass)
    hss = np.zeros(nclass)
    tp = np.zeros(nclass)
    fn = np.zeros(nclass)
    fp = np.zeros(nclass)
    tn = np.zeros(nclass)
    for p in range(nclass):
        tp[p] = confusion_matrix[p][p]
        for q in range(nclass):
            if q != p:
                fn[p] += confusion_matrix[p][q]
                fp[p] += confusion_matrix[q][p]
        tn[p] = N - tp[p] - fn[p] - fp[p]

        recall[p] = round(float(tp[p]) / float(tp[p] + fn[p] + 1e-6), 3)
        precision[p] = round(float(tp[p]) / float(tp[p] + fp[p] + 1e-6), 3)
        accuracy[p] = round(float(tp[p] + tn[p]) / float(N), 3)
        bacc[p] = round(
            0.5 * (float(tp[p]) / float(tp[p] + fn[p]) + float(tn[p]) / float(tn[p] + fp[p])), 3)
        hss[p] = round(2 * float(tp[p] * tn[p] - fp[p] * fn[p])
                       / float((tp[p] + fn[p]) * (fn[p] + tn[p])
                               + (tp[p] + fp[p]) * (fp[p] + tn[p])), 3)
        tss[p] = round(
            (float(tp[p]) / float(tp[p] + fn[p] + 1e-6) - float(fp[p]) / float(fp[p] + tn[p] + 1e-6)), 3)

    # print("tss: " + str(tss))
    # print("hss: " + str(hss))
    # print("bacc: " + str(bacc))
    # print("accuracy: " + str(accuracy))
    # print("precision: " + str(precision))
    # print("recall: " + str(recall))

    return tss[0]
    # return recall, precision, accuracy, bacc, tss, hss, tp, fn, fp, tn


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, rnn_module='LSTM'):
        super(LSTMModel, self).__init__()
        self.input_dim = input_dim
        self.rnn_module = rnn_module
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        # Building your LSTM
        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        if rnn_module == "RNN":
            self.rnn = nn.RNN(input_size=input_dim, hidden_size=hidden_dim, num_layers=layer_dim, dropout=args.dropout,
                              batch_first=True)
        elif rnn_module == "LSTM":
            self.rnn = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=layer_dim, dropout=args.dropout,
                               batch_first=True)
        elif rnn_module == "GRU":
            self.rnn = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=layer_dim, dropout=args.dropout,
                              batch_first=True)

        # rough attention layer
        self.fc_att = nn.Linear(hidden_dim, 1).to(device)
        self.fc0 = nn.Linear(hidden_dim, hidden_dim).to(device)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(args.dropout)
        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # # 28 time steps
        # # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # # If we don't, we'll backprop all the way to the start even after going through another batch
        # # shape of lstm_out: [input_size, batch_size, hidden_dim]
        # out, (hn, cn) = self.rnn(x, (h0.detach(), c0.detach()))
        #
        # # Index hidden state of last time step
        # # out.size() --> 100, 28, 100
        # # out[:, -1, :] --> 100, 100 --> just want last time step hidden states!
        # out = self.fc(out[:, -1, :])
        # # out.size() --> 100, 10

        if self.rnn_module == "RNN":
            h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
            out, (hn) = self.rnn(x, (h0.detach()))
            out = self.fc(out[:, -1, :])
        elif self.rnn_module == "LSTM":
            h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
            c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
            out, (hn, cn) = self.rnn(x, (h0.detach(), c0.detach()))
            out = self.fc(out[:, -1, :])

        elif self.rnn_module == "GRU":
            h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
            out, (hn) = self.rnn(x, (h0.detach())) #-> [batch, layers, hiddendim]
            att = self.fc_att(out).squeeze(-1) # -> [batch, layers]
            att = F.softmax(att, dim=-1)
            r_att = torch.sum(att.unsqueeze(-1) * out, dim=1)  # -> [batch, hiddendim]
            f = self.drop(self.act(self.fc0(out)))  # -> [batch, layers, hiddendim]
            out = self.fc(out[:, -1, :])

        return out

    def initHidden(self, batch_size):
        # initialize hidden state to zeros
        if self.rnn_module == "LSTM":
            return torch.zeros(self.layer_dim, batch_size, self.hidden_dim).to(device),\
                   torch.zeros(self.layer_dim, batch_size, self.hidden_dim).to(device)
        else:
            return torch.zeros(self.layer_dim, batch_size, self.hidden_dim).to(device)


def train(model, device, train_loader, optimizer, epoch, criterion):
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
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

    loss_epoch /= len(train_loader.dataset)
    print("Training Scores:")
    recall, precision, accuracy, bacc, tss, hss, tp, fn, fp, tn = calculate_metrics(confusion_matrix)

    wandb.log({
        "Training_Accuracy": accuracy[0],
        "Training_TSS": tss[0],
        "Training_HSS": hss[0],
        "Training_BACC": bacc[0],
        "Training_Precision": precision[0],
        "Training_Recall": recall[0],
        "Training_Loss": loss_epoch}, step=epoch)


def validate(model, device, valid_loader, criterion, epoch, best_tss, best_epoch):
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
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        valid_loss, correct, len(valid_loader.dataset),
        100. * correct / len(valid_loader.dataset)))

    # # validation conf matrix
    # print(confusion_matrix)
    # # per class accuracy
    # print(confusion_matrix.diag() / confusion_matrix.sum(1))

    print("Validation Scores:")
    recall, precision, accuracy, bacc, tss, hss, tp, fn, fp, tn = calculate_metrics(confusion_matrix)

    wandb.log({
        "Validation_Accuracy": accuracy[0],
        "Validation_TSS": tss[0],
        "Validation_HSS": hss[0],
        "Validation_BACC": bacc[0],
        "Validation_Precision": precision[0],
        "Validation_Recall": recall[0],
        "Validation_Loss": valid_loss}, step=epoch)

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
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    print("Testing Scores:")
    recall, precision, accuracy, bacc, tss, hss, tp, fn, fp, tn = calculate_metrics(confusion_matrix)

    wandb.log({
        "Test_Accuracy": accuracy[0],
        "Test_TSS": tss[0],
        "Test_HSS": hss[0],
        "Test_BACC": bacc[0],
        "Test_Precision": precision[0],
        "Test_Recall": recall[0],
        "Test_Loss": test_loss})


if __name__ == '__main__':
    wandb.init(project='Liu_pytorch')

    # parse hyperparameters
    parser = argparse.ArgumentParser(description='Deep Flare Prediction')
    parser.add_argument('--epochs', type=int, default=40, metavar='N',
                        help='number of epochs to train (default: 40)')
    parser.add_argument('--flare_label', default="M5",
                        help='Types of flare class (default: M5-Class')
    parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 512)')
    parser.add_argument('--learning_rate', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--layer_dim', type=int, default=10, metavar='N',
                        help='how many hidden layers (default: 5)')
    parser.add_argument('--hidden_dim', type=int, default=24, metavar='N',
                        help='how many nodes in layers (default: 64)')
    parser.add_argument('--dropout', type=float, default=0.5, metavar='M',
                        help='percentage dropout (default: 0.5)')
    parser.add_argument('--weight_decay', type=float, default=0.0, metavar='LR',
                        help='L2 regularizing (default: 0.0001)')
    parser.add_argument('--rnn_module', default="LSTM",
                        help='Types of rnn (default: LSTM')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log_interval', type=int, default=30, metavar='N',
                        help='how many batches to wait before logging training status')
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
        n_features = 20
    elif args.flare_label == 'M':
        n_features = 22
    elif args.flare_label == 'C':
        n_features = 14

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

    # set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    np.random.seed(args.seed)

    # setup dataloaders
    X_train_data, y_train_data = load_data(datafile=filepath + 'normalized_training.csv',
                                           flare_label=args.flare_label, series_len=args.layer_dim,
                                           start_feature=start_feature, n_features=n_features,
                                           mask_value=mask_value)

    X_valid_data, y_valid_data = load_data(datafile=filepath + 'normalized_validation.csv',
                                           flare_label=args.flare_label, series_len=args.layer_dim,
                                           start_feature=start_feature, n_features=n_features,
                                           mask_value=mask_value)

    X_test_data, y_test_data = load_data(datafile=filepath + 'normalized_testing.csv',
                                         flare_label=args.flare_label, series_len=args.layer_dim,
                                         start_feature=start_feature, n_features=n_features,
                                         mask_value=mask_value)

    y_train_tr = label_transform(y_train_data)
    y_valid_tr = label_transform(y_valid_data)
    y_test_tr = label_transform(y_test_data)

    # ready custom dataset
    datasets = {}
    datasets['train'] = preprocess_customdataset(X_train_data, y_train_tr)
    datasets['valid'] = preprocess_customdataset(X_valid_data, y_valid_tr)
    datasets['test'] = preprocess_customdataset(X_test_data, y_test_tr)

    train_loader = torch.utils.data.DataLoader(datasets['train'], args.batch_size,
                                               shuffle=False, drop_last=False)
    valid_loader = torch.utils.data.DataLoader(datasets['valid'], args.batch_size,
                                               shuffle=False, drop_last=False)
    test_loader = torch.utils.data.DataLoader(datasets['test'], args.batch_size,
                                              shuffle=False, drop_last=False)

    # make model
    device = torch.device("cuda" if use_cuda else "cpu")
    model = LSTMModel(n_features, hidden_dim=args.hidden_dim, layer_dim=args.layer_dim,
                      output_dim=nclass, rnn_module=args.rnn_module).to(device)

    # restore a previous model
    if args.restore:
        # wandb.restore('model.pt') # todo specify path to restore
        model.load_state_dict(torch.load('./model.pt'))

    try:
        wandb.watch(model, log='all')
    except:
        pass

    # optimizers
    class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train_data), y_train_data)

    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(device))  # weighted cross entropy
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate,
                                 weight_decay=args.weight_decay, amsgrad=False)

    # print model parameters
    print(len(list(model.parameters())))
    for i in range(len(list(model.parameters()))):
        print(list(model.parameters())[i].size())

    # early stopping check
    early_stop = early_stopping.EarlyStopping(mode='max', patience=10)
    best_tss = 0.0
    best_epoch = 0

    print("Training Network...")
    if args.training:
        for epoch in range(args.epochs):
            train(model, device, train_loader, optimizer, epoch, criterion)
            tss, best_tss, best_epoch = validate(model, device, valid_loader, criterion, epoch, best_tss, best_epoch)
            if early_stop.step(tss):
                break

    wandb.log({"Best_Validation_TSS": best_tss,
               "Best_Validation_epoch": best_epoch})

    # reload best tss checkpoint and test
    print("[INFO] Reverting to checkpoint at epoch:" + str(best_epoch))
    model.load_state_dict(torch.load(os.path.join(wandb.run.dir, 'model.pt')))
    test(model, device, test_loader, criterion)

    '''
    K-fold Cross validation
    '''
    # print("Do K-Fold cross validation in skorch wrapper")
    #
    # inputs = torch.tensor(X_train_data).float()
    # labels = torch.tensor(y_train_tr).long()
    # X_valid_data = torch.tensor(X_valid_data).float()
    # y_valid_tr = torch.tensor(y_valid_tr).long()
    #
    # inputs = inputs.numpy()
    # labels = labels.numpy()
    # X_valid_data = X_valid_data.numpy()
    # y_valid_tr = y_valid_tr.numpy()
    #
    # valid_ds = Dataset(X_valid_data, y_valid_tr)
    #
    # tss = EpochScoring(scoring=make_scorer(get_tss), lower_is_better=False, name='tss', use_caching=True)
    # earlystop = EarlyStopping(monitor='tss', lower_is_better=False, patience=10)
    # checkpoint = Checkpoint(monitor='tss_best', dirname='./saved/models/exp1')
    #
    # net = NeuralNetClassifier(
    #     model,
    #     max_epochs=args.epochs,
    #     batch_size=args.batch_size,
    #     criterion=nn.CrossEntropyLoss,
    #     criterion__weight=torch.FloatTensor(class_weights).to(device),
    #     optimizer=torch.optim.Adam,
    #     optimizer__lr=args.learning_rate,
    #     optimizer__weight_decay=args.weight_decay,
    #     optimizer__amsgrad=False,
    #     device=device,
    #     # train_split=None, #die breek die logs
    #     train_split=predefined_split(valid_ds),
    #     callbacks=[tss, earlystop, checkpoint],
    #     # iterator_train__shuffle=True,  # batches shuffle
    #     # warm_start=False
    # )
    #
    # # net.fit(inputs, labels)
    #
    # net.initialize()
    # # net.load_params(checkpoint=checkpoint)  # Select best TSS epoch
    #
    # # inputs = torch.tensor(X_test_data).float()
    # # labels = torch.tensor(y_test_tr).long()
    # #
    # # inputs = inputs.numpy()
    # # labels = labels.numpy()
    #
    # # net.max_epochs = 0
    # score = cross_val_score(net, inputs, labels, cv=2, scoring=make_scorer(get_tss))
    # print(score)
    # # y_pred = cross_val_predict(net, inputs, labels, cv=2)
    # # print(y_pred)
    #
    # '''
    # Test Results
    # '''
    # # inputs = torch.tensor(X_valid_data).float()
    # # labels = torch.tensor(y_valid_tr).long()
    # inputs = torch.tensor(X_test_data).float()
    # labels = torch.tensor(y_test_tr).long()
    #
    # inputs = inputs.numpy()
    # labels = labels.numpy()
    #
    # y_test = net.predict(inputs)
    # tss_test_score = get_tss(labels, y_test)
    # print("Test TSS:" + str(tss_test_score))

    # Save model to W&B
    # torch.save(model.state_dict(), os.path.join(wandb.run.dir, 'model.pt'))
    print('Finished')
