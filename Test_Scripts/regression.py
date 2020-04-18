from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from skorch import NeuralNetRegressor
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from skorch.dataset import Dataset
from sklearn.preprocessing import MinMaxScaler


def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(1, L+1):
        train_seq = []
        if i <= tw:
            train_seq = np.zeros((tw-i,n_features))
            train_seq = np.concatenate((train_seq, input_data[0:i]), axis=0)
        elif i > tw:
            train_seq = input_data[i-tw:i]
        inout_seq.append(train_seq)
    return inout_seq


class LSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, batch_size, output_dim=1,
                 num_layers=2):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers

        # Define the LSTM layer
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim,
                            self.num_layers, batch_first=True)

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
                         self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, input.size(0),
                         self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.lstm(input, (h0, c0))
        out = self.linear(out[:, -1, :])
        return out


class RegressorModule(nn.Module):
    def __init__(self, num_units=10, nonlin=F.relu, ):
        super(RegressorModule, self).__init__()
        self.num_units = num_units
        self.nonlin = nonlin

        self.dense0 = nn.Linear(20, num_units)
        self.nonlin = nonlin
        self.dense1 = nn.Linear(num_units, 10)
        self.output = nn.Linear(10, 1)

    def forward(self, X, **kwargs):
        X = self.nonlin(self.dense0(X))
        X = F.relu(self.dense1(X))
        X = self.output(X)
        return X


# Network params
n_features = 20
batch_size = 32
# If `per_element` is True, then LSTM reads in one timestep at a time.
per_element = False
if per_element:
    lstm_input_size = 1
else:
    lstm_input_size = n_features
# size of hidden layers
h1 = 64
output_dim = 1
num_layers = 2
learning_rate = 1e-1
num_epochs = 300
dtype = torch.float
train_window = 12

X, y = make_regression(1000, n_features, n_informative=2, random_state=0,
                       shuffle=False)
X = X.astype(np.float32)
y = y.astype(np.float32) / 100
y = y.reshape(-1, 1)

X = X.astype(np.float32)
ds = Dataset(X, y)
y = np.array([y for _, y in iter(ds)])
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8,
                                                    random_state=1)
# Normalize
scaler = MinMaxScaler(feature_range=(-1, 1))
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# convert to sequences
X_train_seq = create_inout_sequences(X_train, train_window)
X_test_seq = create_inout_sequences(X_test, train_window)

# model = NeuralNetRegressor(RegressorModule, max_epochs=20, lr=0.1,
#     device='cuda',  # uncomment this to train with CUDA
# )
model = LSTM(lstm_input_size, h1, batch_size=batch_size, output_dim=output_dim,
             num_layers=num_layers)

net = NeuralNetRegressor(model, lr=learning_rate, max_epochs=num_epochs,
                         criterion=nn.MSELoss, batch_size=32,
                         # callbacks=[bacc, train_acc, roc_auc,
                         #            EarlyStopping(monitor='train_bacc',
                         #                          lower_is_better=False,
                         #                          patience=100), checkpoint],
                         warm_start=False)
X_test_seq = torch.tensor(X_test_seq).float()
X_test_seq = X_test_seq.numpy()
X_train_seq = torch.tensor(X_train_seq).float()
X_train_seq = X_train_seq.numpy()
net.fit(X_train_seq, y_train)

y_train_pred = net.predict(X_train_seq)
y_pred = net.predict(X_test_seq)

# Plot prediction
plt.plot(np.linspace(0, len(y_train), len(y_train)), y_train)
plt.plot(np.linspace(len(y_train), len(y_train) + len(y_test), len(y_test)),
         y_test)
plt.plot(np.linspace(len(y_train), len(y_train) + len(y_test), len(y_test)),
         y_pred)
plt.plot(np.linspace(0, len(y_train), len(y_train)), y_train_pred)

plt.show()
