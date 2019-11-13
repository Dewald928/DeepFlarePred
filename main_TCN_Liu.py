'''
Ecclesiastes 5:12 New King James Version (NKJV)
12 The sleep of a laboring man is sweet,
Whether he eats little or much;
But the abundance of the rich will not permit him to sleep.
'''
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight

import sys
import os
import numpy as np
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from Test_Scripts.tcn import TemporalConvNet

# import skorch
# from skorch import NeuralNetClassifier
# from sklearn.model_selection import cross_val_predict

from data_loader import CustomDataset
from utils import early_stopping
import wandb

from captum.attr import IntegratedGradients
from captum.attr import DeepLift
from captum.attr import DeepLiftShap
from captum.attr import Saliency
from captum.attr import GradientAttribution

import matplotlib.pyplot as plt
from scipy import stats

# import shap


def load_data(datafile, flare_label, series_len, start_feature, n_features,
              mask_value):
    df = pd.read_csv(datafile)
    df_values = df.values
    feature_names = list(df.columns)
    X = []
    y = []
    tmp = []
    for k in range(start_feature, start_feature + n_features):
        tmp.append(mask_value)
    for idx in range(0, len(df_values)):
        each_series_data = []
        row = df_values[idx]
        if flare_label == 'M5' and row[1][0] == 'M' and float(
                row[1][1:]) >= 5.0:
            label = 'X'
        else:
            label = row[1][0]
        if flare_label == 'M' and label == 'X':
            label = 'M'
        if flare_label == 'C' and (label == 'X' or label == 'M'):
            label = 'C'
        if flare_label == 'B' and (
                label == 'X' or label == 'M' or label == 'C'):
            label = 'B'
        if flare_label == 'M5' and (
                label == 'M' or label == 'C' or label == 'B'):
            label = 'N'
        if flare_label == 'M' and (label == 'B' or label == 'C'):
            label = 'N'
        if flare_label == 'C' and label == 'B':
            label = 'N'
        has_zero_record = False
        # if at least one of the 25 physical feature values is missing,
        # then discard it.
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
            each_series_data.append(
                row[start_feature:start_feature + n_features].tolist())
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

                if len(
                        each_series_data) < series_len and \
                        has_zero_record_tmp is True:
                    each_series_data.insert(0, tmp)

                if len(
                        each_series_data) < series_len and \
                        has_zero_record_tmp is False:
                    each_series_data.insert(0, prev_row[
                                               start_feature:start_feature +
                                                             n_features].tolist())
                itr_idx -= 1

            while len(each_series_data) > 0 and len(
                    each_series_data) < series_len:
                each_series_data.insert(0, tmp)

            if len(each_series_data) > 0:
                X.append(np.array(each_series_data).reshape(series_len,
                                                            n_features).tolist())
                y.append(label)
    X_arr = np.array(X)
    y_arr = np.array(y)
    print(X_arr.shape)
    return X_arr, y_arr


def get_feature_names(datafile):
    df = pd.read_csv(datafile)
    feature_names = list(df.columns)
    return feature_names


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
        bacc[p] = round(0.5 * (
                float(tp[p]) / float(tp[p] + fn[p]) + float(tn[p]) / float(
            tn[p] + fp[p])), 3)
        hss[p] = round(2 * float(tp[p] * tn[p] - fp[p] * fn[p]) / float(
            (tp[p] + fn[p]) * (fn[p] + tn[p]) + (tp[p] + fp[p]) * (
                    fp[p] + tn[p])), 3)
        tss[p] = round((float(tp[p]) / float(tp[p] + fn[p] + 1e-6) - float(
            fp[p]) / float(fp[p] + tn[p] + 1e-6)), 3)

    print("tss: " + str(tss))
    print("hss: " + str(hss))
    print("bacc: " + str(bacc))
    print("accuracy: " + str(accuracy))
    print("precision: " + str(precision))
    print("recall: " + str(recall))

    return recall, precision, accuracy, bacc, tss, hss, tp, fn, fp, tn


class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size,
                 dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels,
                                   kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size).to(device)
        self.init_weights()

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x):
        y1 = self.tcn(x)
        return self.linear(y1[:, :, -1])


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim,
                 rnn_module='LSTM'):
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
            self.rnn = nn.RNN(input_size=input_dim, hidden_size=hidden_dim,
                              num_layers=layer_dim, dropout=args.dropout,
                              batch_first=True)
        elif rnn_module == "LSTM":
            self.rnn = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim,
                               num_layers=layer_dim, dropout=args.dropout,
                               batch_first=True)
        elif rnn_module == "GRU":
            self.rnn = nn.GRU(input_size=input_dim, hidden_size=hidden_dim,
                              num_layers=layer_dim, dropout=args.dropout,
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
        # # We need to detach as we are doing truncated backpropagation
        # through time (BPTT)
        # # If we don't, we'll backprop all the way to the start even after
        # going through another batch
        # # shape of lstm_out: [input_size, batch_size, hidden_dim]
        # out, (hn, cn) = self.rnn(x, (h0.detach(), c0.detach()))
        #
        # # Index hidden state of last time step
        # # out.size() --> 100, 28, 100
        # # out[:, -1, :] --> 100, 100 --> just want last time step hidden
        # states!
        # out = self.fc(out[:, -1, :])
        # # out.size() --> 100, 10

        if self.rnn_module == "RNN":
            h0 = torch.zeros(self.layer_dim, x.size(0),
                             self.hidden_dim).requires_grad_().to(device)
            out, (hn) = self.rnn(x, (h0.detach()))
            out = self.fc(out[:, -1, :])
        elif self.rnn_module == "LSTM":
            h0 = torch.zeros(self.layer_dim, x.size(0),
                             self.hidden_dim).requires_grad_().to(device)
            c0 = torch.zeros(self.layer_dim, x.size(0),
                             self.hidden_dim).requires_grad_().to(device)
            out, (hn, cn) = self.rnn(x, (h0.detach(), c0.detach()))
            out = self.fc(out[:, -1, :])

        elif self.rnn_module == "GRU":
            h0 = torch.zeros(self.layer_dim, x.size(0),
                             self.hidden_dim).requires_grad_().to(device)
            out, (hn) = self.rnn(x, (
                h0.detach()))  # -> [batch, layers, hiddendim]
            att = self.fc_att(out).squeeze(-1)  # -> [batch, layers]
            att = F.softmax(att, dim=-1)
            r_att = torch.sum(att.unsqueeze(-1) * out,
                              dim=1)  # -> [batch, hiddendim]
            f = self.drop(
                self.act(self.fc0(out)))  # -> [batch, layers, hiddendim]
            out = self.fc(out[:, -1, :])

        return out

    def initHidden(self, batch_size):
        # initialize hidden state to zeros
        if self.rnn_module == "LSTM":
            return torch.zeros(self.layer_dim, batch_size, self.hidden_dim).to(
                device), torch.zeros(self.layer_dim, batch_size,
                                     self.hidden_dim).to(device)
        else:
            return torch.zeros(self.layer_dim, batch_size, self.hidden_dim).to(
                device)


def train(model, device, train_loader, optimizer, epoch, criterion):
    model.train()
    confusion_matrix = torch.zeros(nclass, nclass)
    loss_epoch = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        try:
            data = data.view(len(data), n_features, args.layer_dim)
        except:
            print("woah the cowboy")
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
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                         batch_idx * len(
                                                                             data),
                                                                         len(
                                                                             train_loader.dataset),
                                                                         100. * batch_idx / len(
                                                                             train_loader),
                                                                         loss.item()))

    loss_epoch /= len(train_loader.dataset)
    print("Training Scores:")
    recall, precision, accuracy, bacc, tss, hss, tp, fn, fp, \
    tn = calculate_metrics(
        confusion_matrix)

    wandb.log({"Training_Accuracy": accuracy[0], "Training_TSS": tss[0],
               "Training_HSS": hss[0], "Training_BACC": bacc[0],
               "Training_Precision": precision[0],
               "Training_Recall": recall[0], "Training_Loss": loss_epoch},
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
            try:
                data = data.view(len(data), n_features, args.layer_dim)
            except:
                print("woah the cowboy")
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
    tn = calculate_metrics(
        confusion_matrix)

    wandb.log({"Validation_Accuracy": accuracy[0], "Validation_TSS": tss[0],
               "Validation_HSS": hss[0], "Validation_BACC": bacc[0],
               "Validation_Precision": precision[0],
               "Validation_Recall": recall[0], "Validation_Loss": valid_loss},
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
            model.to(device)
            data, target = data.to(device), target.to(device)
            try:
                data = data.view(len(data), n_features, args.layer_dim)
            except:
                print("woah the cowboy")
            output = model(data).to(device)
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
    tn = calculate_metrics(
        confusion_matrix)

    wandb.log(
        {"Test_Accuracy": accuracy[0], "Test_TSS": tss[0], "Test_HSS": hss[0],
         "Test_BACC": bacc[0], "Test_Precision": precision[0],
         "Test_Recall": recall[0], "Test_Loss": test_loss})


def interpret_model(model, device, test_loader):
    print("\n Interpreting Model...")
    attr_ig = []
    attr_ig_avg = []
    attr_sal = []
    attr_sal_avg = []
    model.eval()
    i = 1
    with torch.no_grad():
        for data, target in test_loader:
            print("Batch#" + str(i))
            model.to(device)
            data, target = data.to(device), target.to(device)
            data = data.view(len(data), n_features, args.layer_dim)

            ig = IntegratedGradients(model.to(device))
            sal = Saliency(model.to(device))  # todo deeplift not working
            temp_data = torch.clone(data).to(device)
            attr_ig = ig.attribute(temp_data, target=1)
            attr_sal = sal.attribute(inputs=temp_data, target=1)
            try:
                attr_ig_avg = attr_ig if i == 1 else (
                                                             attr_ig_avg +
                                                             attr_ig) / i
                attr_sal_avg = attr_sal if i == 1 else (
                                                               attr_sal_avg
                                                               + attr_sal) / i
            except:
                print("hmmmm???")
                pass
            attr_ig = attr_ig.cpu().detach().numpy()
            attr_sal = attr_sal.cpu().detach().numpy()

            if i == len(test_loader) - 1:
                break
            i += 1
    attr_ig_avg = attr_ig_avg.cpu().detach().numpy()
    attr_sal_avg = attr_sal_avg.cpu().detach().numpy()

    return attr_ig, attr_sal, attr_ig_avg, attr_sal_avg


# Helper method to print importance and visualize distribution
def visualize_importance(feature_names, importances, std,
                         title="Average Feature Importance", plot=True,
                         axis_title="Features"):
    # print(title)
    # for i in range(len(feature_names)):
    #     print(feature_names[i], ": ", '%.3f' % (importances[i]))
    x_pos = (np.arange(len(feature_names)))
    if plot:
        fig = plt.figure(figsize=(8, 4))
        plt.bar(x_pos, importances.reshape(n_features),
                yerr=std.reshape(n_features), align='center')
        plt.xticks(x_pos, feature_names, wrap=False, rotation=60)
        plt.xlabel(axis_title)
        plt.title(title)
        fig.show()
        wandb.log({'Image of features': wandb.Image(fig)})

        # plot ranked
        df_sorted = pd.DataFrame({'Features': feature_names,
                                  "Importances": importances.reshape(
                                      n_features)})
        df_sorted = df_sorted.sort_values('Importances')
        fig_sorted = df_sorted.plot(kind='bar', y='Importances', x='Features',
                                    title=title, figsize=(8, 4),
                                    yerr=std.reshape(n_features))
        plt.show()
        wandb.log({'Feature Ranking': wandb.Image(fig_sorted)})


def check_significance(attr, test_features, feature_num=1):
    fig = plt.hist(attr[:, feature_num], 100)
    plt.title("Distribution of Attribution Values")
    plt.show()

    bin_means, bin_edges, _ = stats.binned_statistic(test_features[:, 1],
                                                     attr[:, 1],
                                                     statistic='mean', bins=6)
    bin_count, _, _ = stats.binned_statistic(test_features[:, 1], attr[:, 1],
                                             statistic='count', bins=6)

    bin_width = (bin_edges[1] - bin_edges[0])
    bin_centers = bin_edges[1:] - bin_width / 2
    plt.scatter(bin_centers, bin_means, s=bin_count)
    plt.xlabel("Average Sibsp Feature Value")
    plt.ylabel("Average Attribution")


if __name__ == '__main__':
    wandb.init(project='Liu_pytorch', notes='TCN')

    # parse hyperparameters
    parser = argparse.ArgumentParser(description='Deep Flare Prediction')
    parser.add_argument('--epochs', type=int, default=10,
                        help='upper epoch limit (default: 100)')
    parser.add_argument('--flare_label', default="M5",
                        help='Types of flare class (default: M-Class')
    parser.add_argument('--batch_size', type=int, default=2048, metavar='N',
                        help='input batch size for training (default: 256)')
    parser.add_argument('--learning_rate', type=float, default=1e-2,
                        help='initial learning rate (default: 1e-3)')
    parser.add_argument('--layer_dim', type=int, default=1, metavar='N',
                        help='how many hidden layers (default: 5)')

    parser.add_argument('--levels', type=int, default=2,
                        help='# of levels (default: 4)')
    parser.add_argument('--ksize', type=int, default=2,
                        help='kernel size (default: 5)')
    parser.add_argument('--nhid', type=int, default=40,
                        help='number of hidden units per layer (default: 128)')

    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')

    parser.add_argument('--dropout', type=float, default=0.0,
                        help='dropout applied to layers (default: 0.25)')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        metavar='LR', help='L2 regularizing (default: 0.0001)')
    parser.add_argument('--rnn_module', default="TCN",
                        help='Types of rnn (default: LSTM')

    parser.add_argument('--clip', type=float, default=0.2,
                        help='gradient clip, -1 means no clip (default: 0.2)')
    parser.add_argument('--optim', type=str, default='Adam',
                        help='optimizer to use (default: Adam)')
    parser.add_argument('--seed', type=int, default=1,
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
        n_features = 40  # 20 original
    elif args.flare_label == 'M':
        n_features = 22
    elif args.flare_label == 'C':
        n_features = 14
    feature_names = get_feature_names(filepath + 'normalized_training.csv')

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
    X_train_data, y_train_data = load_data(
        datafile=filepath + 'normalized_training.csv',
        flare_label=args.flare_label, series_len=args.layer_dim,
        start_feature=start_feature, n_features=n_features,
        mask_value=mask_value)

    X_valid_data, y_valid_data = load_data(
        datafile=filepath + 'normalized_validation.csv',
        flare_label=args.flare_label, series_len=args.layer_dim,
        start_feature=start_feature, n_features=n_features,
        mask_value=mask_value)

    X_test_data, y_test_data = load_data(
        datafile=filepath + 'normalized_testing.csv',
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

    device = torch.device("cuda" if use_cuda else "cpu")
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
    print(len(list(model.parameters())))
    for i in range(len(list(model.parameters()))):
        print(list(model.parameters())[i].size())

    # early stopping check
    early_stop = early_stopping.EarlyStopping(mode='max', patience=30)
    best_tss = 0.0
    best_epoch = 0

    print("Training Network...")
    if args.training:
        for epoch in range(args.epochs):
            train(model, device, train_loader, optimizer, epoch, criterion)
            tss, best_tss, best_epoch = validate(model, device, valid_loader,
                                                 criterion, epoch, best_tss,
                                                 best_epoch)
            if early_stop.step(tss) and args.early_stop:
                break

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

    # Model interpretation
    attr_ig, attr_sal, attr_ig_avg, attr_sal_avg = interpret_model(model,
                                                                   device,
                                                                   test_loader)
    visualize_importance(
        np.array(feature_names[start_feature:start_feature + n_features]),
        np.mean(attr_ig_avg, axis=0), np.std(attr_ig_avg, axis=0),
        title="Integrated Gradient Features")
    visualize_importance(
        np.array(feature_names[start_feature:start_feature + n_features]),
        np.mean(attr_sal_avg, axis=0), np.std(attr_sal_avg, axis=0),
        title="Saliency Features")

    print('Finished')
