import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, device,
                 rnn_module='LSTM', dropout=0.4):
        super(LSTMModel, self).__init__()
        self.device = device
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
                              num_layers=layer_dim, dropout=dropout,
                              batch_first=True)
        elif rnn_module == "LSTM":
            self.rnn = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim,
                               num_layers=layer_dim, dropout=dropout,
                               batch_first=True)
        elif rnn_module == "GRU":
            self.rnn = nn.GRU(input_size=input_dim, hidden_size=hidden_dim,
                              num_layers=layer_dim, dropout=dropout,
                              batch_first=True)

        # rough attention layer
        self.fc_att = nn.Linear(hidden_dim, 1).to(self.device)
        self.fc0 = nn.Linear(hidden_dim, hidden_dim).to(self.device)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
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
                             self.hidden_dim).requires_grad_().to(self.device)
            out, (hn) = self.rnn(x, (h0.detach()))
            out = self.fc(out[:, -1, :])
        elif self.rnn_module == "LSTM":
            h0 = torch.zeros(self.layer_dim, x.size(0),
                             self.hidden_dim).requires_grad_().to(self.device)
            c0 = torch.zeros(self.layer_dim, x.size(0),
                             self.hidden_dim).requires_grad_().to(self.device)
            out, (hn, cn) = self.rnn(x, (h0.detach(), c0.detach()))
            out = self.fc(out[:, -1, :])

        elif self.rnn_module == "GRU":
            h0 = torch.zeros(self.layer_dim, x.size(0),
                             self.hidden_dim).requires_grad_().to(self.device)
            out, (hn) = self.rnn(x, (h0.detach()))  # -> [batch, layers,
            # hiddendim]
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
                                     self.hidden_dim).to(self.device)
        else:
            return torch.zeros(self.layer_dim, batch_size, self.hidden_dim).to(
                self.device)