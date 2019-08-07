import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix
import sys
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

import CustomDataset

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


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        # Building your LSTM
        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # Initialize cell state
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # 28 time steps
        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Index hidden state of last time step
        # out.size() --> 100, 28, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states!
        out = self.fc(out[:, -1, :])
        # out.size() --> 100, 10
        return out


if __name__ == '__main__':
    flare_label = sys.argv[1]
    filepath = './Data/Lui/' + flare_label + '/'
    num_of_fold = 10
    n_features = 0
    if flare_label == 'M5':
        n_features = 20
    elif flare_label == 'M':
        n_features = 22
    elif flare_label == 'C':
        n_features = 14
    start_feature = 5
    mask_value = 0
    series_len = 10
    epochs = 7
    batch_size = 256
    learning_rate = 1e-3
    nclass = 2
    hidden_dim = 24
    thlistsize = 201
    thlist = np.linspace(0, 1, thlistsize)

    X_train_data, y_train_data = load_data(datafile=filepath + 'normalized_training.csv',
                                           flare_label=flare_label, series_len=series_len,
                                           start_feature=start_feature, n_features=n_features,
                                           mask_value=mask_value)

    X_valid_data, y_valid_data = load_data(datafile=filepath + 'normalized_validation.csv',
                                           flare_label=flare_label, series_len=series_len,
                                           start_feature=start_feature, n_features=n_features,
                                           mask_value=mask_value)

    X_test_data, y_test_data = load_data(datafile=filepath + 'normalized_testing.csv',
                                         flare_label=flare_label, series_len=series_len,
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

    train_loader = torch.utils.data.DataLoader(datasets['train'], batch_size,
                                               shuffle=False, drop_last=False)
    valid_loader = torch.utils.data.DataLoader(datasets['valid'], batch_size,
                                               shuffle=False, drop_last=False)
    test_loader = torch.utils.data.DataLoader(datasets['test'], batch_size,
                                              shuffle=False, drop_last=False)

    # make model
    model = LSTMModel(n_features, hidden_dim=hidden_dim, layer_dim=series_len, output_dim=nclass)

    #optimizers
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print(len(list(model.parameters())))

    for i in range(len(list(model.parameters()))):
        print(list(model.parameters())[i].size())


    iter = 0
    for epoch in range(epochs):
        for i, (data, labels) in enumerate(train_loader):
            # Load data as a torch tensor with gradient accumulation abilities
            data = data.view(-1, series_len, n_features).requires_grad_()
            # labels = labels.resize_((256,2))

            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()

            # Forward pass to get output/logits
            # outputs.size() --> 100, 10
            outputs = model(data)

            # Calculate Loss: softmax --> cross entropy loss
            loss = criterion(outputs, labels)

            # Getting gradients w.r.t. parameters
            loss.backward()

            # Updating parameters
            optimizer.step()

            iter += 1

            if iter % 500 == 0:
                # Calculate Accuracy         
                correct = 0
                total = 0
                # Iterate through test dataset
                for data, labels in test_loader:
                    # Resize data
                    data = data.view(-1, seq_dim, n_features)

                    # Forward pass only to get logits/output
                    outputs = model(data)

                    # Get predictions from the maximum value
                    _, predicted = torch.max(outputs.data, 1)

                    # Total number of labels
                    total += labels.size(0)

                    # Total correct predictions
                    correct += (predicted == labels).sum()

                accuracy = 100 * correct / total

                # Print Loss
                print('Iteration: {}. Loss: {}. Accuracy: {}'.format(iter, loss.item(), accuracy))



    #test model



