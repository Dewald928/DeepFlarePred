import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
import sys
import numpy as np
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from skorch import NeuralNetClassifier
from sklearn.model_selection import cross_val_predict

from data_loader import CustomDataset
import wandb


"""
Dataloader
"""
def load_data(datafile, flare_label, series_len, start_feature, n_features, mask_value):
    df = pd.read_csv(datafile)
    df_values = df.values
    X = []
    y = []
    tmp = []
    X = df_values[:, start_feature:start_feature+n_features]
    y = df_values[:,0]
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


"""
MLP model
"""


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5,):
        super(MLP, self).__init__()
        self.dropout = nn.Dropout(dropout)

        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, X, **kwargs):
        X = F.relu(self.hidden(X))
        X = self.dropout(X)
        X = F.softmax(self.output(X), dim=-1)
        return X

"""
Skorch
"""
if __name__ == '__main__':
    wandb.init(project='Liu_MLP')

    # parse hyperparameters
    parser = argparse.ArgumentParser(description='Deep Flare Prediction')
    parser.add_argument('--batch_size', type=int, default=512, metavar='N',
                        help='input batch size for training (default: 256)')
    parser.add_argument('--test_batch_size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=40, metavar='N',
                        help='number of epochs to train (default: 15)')
    parser.add_argument('--learning_rate', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log_interval', type=int, default=30, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--flare_label', default="M",
                        help='Types of flare class (default: M-Class')
    parser.add_argument('--layer_dim', type=int, default=5, metavar='N',
                        help='how many hidden layers (default: 5)')
    parser.add_argument('--hidden_dim', type=int, default=64, metavar='N',
                        help='how many nodes in layers (default: 64)')
    parser.add_argument('--dropout', type=float, default=0.4, metavar='M',
                        help='percentage dropout (default: 0.4)')
    parser.add_argument('--weight_decay', type=float, default=0.0001, metavar='LR',
                        help='L2 regularizing (default: 0.0001)')
    parser.add_argument('--rnn_module', default="GRU",
                        help='Types of rnn (default: LSTM')
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
    torch.backends.cudnn.benchmark = False  # And this
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

    device = torch.device("cuda" if use_cuda else "cpu")
    model = MLP(input_dim=n_features, hidden_dim=args.hidden_dim, output_dim=nclass).to(device)

    net = NeuralNetClassifier(
        model,
        max_epochs=20,
        lr=0.1,
        device=device,
    )

    net.fit(X_train_data.astype(np.float32), y_train_tr.astype(np.long))

    y_pred = net.predict(X_train_data[:5])
    print(y_pred)

    print("finished")