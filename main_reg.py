import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import wandb
import yaml
import sklearn
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from skorch import NeuralNetRegressor
from skorch.dataset import Dataset
from skorch.helper import predefined_split
from skorch.callbacks import *

from data_loader import data_loader


class LSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, batch_size, output_dim=1,
                 num_layers=2):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers

        # Define the LSTM layer
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers,
                            batch_first=True, dropout=cfg.dropout)

        # Define the output layer
        self.linear = nn.Linear(self.hidden_dim, output_dim)

    def init_hidden(self):
        # This is what we'll initialise our hidden state as
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))

    def forward(self, input):
        # Forward pass through LSTM layer
        # shape of lstm_out: [input_size, batch_size, hidden_dim]
        # shape of self.hidden: (a, b), where a and b both
        # have shape (num_layers, batch_size, hidden_dim).
        # lstm_out, self.hidden = self.lstm(
        #     input.view(len(input), self.batch_size, -1))
        #
        # # Only take the output from the final timetep
        # # Can pass on the entirety of lstm_out to the next layer if it is a
        # # seq2seq prediction
        # y_pred = self.linear(lstm_out[-1].view(self.batch_size, -1))

        h0 = torch.zeros(self.num_layers, input.size(0),
                         self.hidden_dim).requires_grad_().to(device)
        c0 = torch.zeros(self.num_layers, input.size(0),
                         self.hidden_dim).requires_grad_().to(device)
        out, (hn, cn) = self.lstm(input, (h0, c0))
        out = self.linear(out[:, -1, :])
        return out


def init_project():
    with open('config-defaults.yaml', 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.Loader)

    model_type = cfg['model_type']['value']
    project = 'liu_regression'
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

    # Initialize dataloader
    X_train_data, y_train_data = data_loader.load_reg_data(
        datafile=filepath + 'normalized_training.csv',
        flare_label=cfg.flare_label, series_len=cfg.seq_len,
        start_feature=start_feature, n_features=cfg.n_features,
        mask_value=mask_value)

    X_valid_data, y_valid_data = data_loader.load_reg_data(
        datafile=filepath + 'normalized_validation.csv',
        flare_label=cfg.flare_label, series_len=cfg.seq_len,
        start_feature=start_feature, n_features=cfg.n_features,
        mask_value=mask_value)

    X_test_data, y_test_data = data_loader.load_reg_data(
        datafile=filepath + 'normalized_testing.csv',
        flare_label=cfg.flare_label, series_len=cfg.seq_len,
        start_feature=start_feature, n_features=cfg.n_features,
        mask_value=mask_value)

    # Network params
    # If `per_element` is True, then LSTM reads in one timestep at a time.
    per_element = False
    if per_element:
        lstm_input_size = 1
    else:
        lstm_input_size = cfg.n_features
    # size of hidden layers
    h1 = 128
    output_dim = 1
    num_layers = cfg.seq_len
    dtype = torch.float

    X_train = torch.tensor(X_train_data).float()
    y_train = y_train_data.reshape(-1, 1)
    y_train = torch.tensor(y_train).float()

    X_valid = torch.tensor(X_valid_data).float()
    y_valid = y_valid_data.reshape(-1, 1)
    y_valid = torch.tensor(y_valid).float()

    X_test = torch.tensor(X_test_data).float()
    y_test = y_test_data.reshape(-1, 1)
    y_test = torch.tensor(y_test).float()

    X_train = X_train.numpy()
    X_valid = X_valid.numpy()
    X_test = X_test.numpy()
    y_train = y_train.numpy()
    y_valid = y_valid.numpy()
    y_test = y_test.numpy()

    # normalize
    scaler = MinMaxScaler()
    y_train = scaler.fit_transform(y_train)
    y_valid = scaler.transform(y_valid)
    y_test = scaler.transform(y_test)

    # valid set
    valid_ds = Dataset(X_valid, y_valid)

    # scoring
    valid_r2 = EpochScoring(scoring='r2', lower_is_better=False,
                            name='valid_r2', use_caching=True)
    train_r2 = EpochScoring(scoring='r2', lower_is_better=False,
                            name='train_r2', use_caching=True, on_train=True)
    train_mse = EpochScoring(scoring='neg_mean_squared_error', on_train=True,
                             lower_is_better=False)

    model = LSTM(lstm_input_size, h1, batch_size=cfg.batch_size,
                 output_dim=output_dim, num_layers=num_layers)

    net = NeuralNetRegressor(model, lr=cfg.learning_rate,
                             max_epochs=cfg.epochs, criterion=nn.MSELoss,
                             batch_size=cfg.batch_size,
                             train_split=predefined_split(valid_ds),
                             callbacks=[train_r2, valid_r2, train_mse],
                             device=device, warm_start=False)

    net.fit(X_train, y_train)

    y_train_pred = net.predict(X_train)
    y_valid_pred = net.predict(X_valid)
    y_test_pred = net.predict(X_test)

    # get real flare values
    y_train_true = scaler.inverse_transform(y_train)
    y_train_pred_true = scaler.inverse_transform(y_train_pred)
    y_valid_true = scaler.inverse_transform(y_valid)
    y_valid_pred_true = scaler.inverse_transform(y_valid_pred)
    y_test_true = scaler.inverse_transform(y_test)
    y_test_pred_true = scaler.inverse_transform(y_test_pred)

    # Plot prediction
    fig = plt.figure()
    plt.plot(np.linspace(0, len(y_train), len(y_train)), y_train_true)
    plt.plot(
        np.linspace(len(y_train), len(y_train) + len(y_valid), len(y_valid)),
        y_valid_true)
    plt.plot(np.linspace(len(y_train) + len(y_valid),
                         len(y_train) + len(y_valid) + len(y_test),
                         len(y_test)), y_test_true)

    plt.plot(np.linspace(0, len(y_train), len(y_train)), y_train_pred_true)
    plt.plot(
        np.linspace(len(y_train), len(y_train) + len(y_valid), len(y_valid)),
        y_valid_pred_true)
    plt.plot(np.linspace(len(y_train) + len(y_valid),
                         len(y_train) + len(y_valid) + len(y_test),
                         len(y_test)), y_test_pred_true)

    plt.yscale('log')
    fig.show()
    wandb.log({'Regression_Plot': wandb.Image(fig)})
