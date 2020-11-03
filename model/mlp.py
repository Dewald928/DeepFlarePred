import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPModule(nn.Module):
    def __init__(
            self,
            input_units=20,
            output_units=2,
            hidden_units=10,
            num_hidden=1,
            nonlin=nn.ReLU(),
            output_nonlin=None,
            dropout=0,
            squeeze_output=False,
    ):
        super().__init__()
        self.input_units = input_units
        self.output_units = output_units
        self.hidden_units = hidden_units
        self.num_hidden = num_hidden
        self.nonlin = nonlin
        self.output_nonlin = output_nonlin
        self.dropout = dropout
        self.squeeze_output = squeeze_output

        self.reset_params()

    def init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            print(m.weight)
            # m.bias.data.fill_(0.01)

    def reset_params(self):
        """(Re)set all parameters."""
        units = [self.input_units]
        units += [self.hidden_units] * self.num_hidden
        units += [self.output_units]

        sequence = []
        for u0, u1 in zip(units, units[1:]):
            sequence.append(nn.Linear(u0, u1, bias=False))
            sequence.append(nn.BatchNorm1d(num_features=u1))
            sequence.append(nn.ReLU())
            sequence.append(nn.Dropout(self.dropout))

        sequence = sequence[:-2]
        if self.output_nonlin:
            sequence.append(self.output_nonlin)

        def init_weights(m):
            if type(m) == nn.Linear:
                torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')

        self.sequential = nn.Sequential(*sequence)
        # self.sequential.apply(init_weights)

    def forward(self, X):  # pylint: disable=arguments-differ
        X = self.sequential(X)
        if self.squeeze_output:
            X = X.squeeze(-1)
        return X


class block1(nn.Module):
    def __init__(self, input_units=20, output_units=2, hidden_units=10,
              num_hidden=1):
        super(block1, self).__init__()
        self.linear1 = nn.Linear(input_units, hidden_units)
        self.linear2 = nn.Linear(hidden_units, hidden_units)
        self.linear3 = nn.Linear(hidden_units, hidden_units)
        self.linear4 = nn.Linear(hidden_units, input_units)
        self.output_units = output_units
        self.batch_norm = nn.BatchNorm1d(input_units)
        self.batch_norm1 = nn.BatchNorm1d(hidden_units)

    def forward(self, x):
        residual = x
        out = F.relu(self.batch_norm1(self.linear1(x)))
        out = F.relu(self.batch_norm1(self.linear2(out)))
        out = F.relu(self.batch_norm1(self.linear3(out)))
        out = F.relu(self.batch_norm(self.linear4(out)))

        out += residual
        return F.relu(out)


class DEFNR(nn.Module):
    def __init__(
            self,
            input_units=20,
            output_units=2,
            hidden_units=10,
            num_hidden=1,
            nonlin=nn.ReLU(),
            output_nonlin=None,
            dropout=0,
            squeeze_output=False,
    ):
        super().__init__()
        self.input_units = input_units
        self.output_units = output_units
        self.hidden_units = hidden_units
        self.num_hidden = num_hidden
        self.nonlin = nonlin
        self.output_nonlin = output_nonlin
        self.dropout = dropout
        self.squeeze_output = squeeze_output
        self.linear = nn.Linear(input_units, output_units)

        self.reset_params()

    def init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            print(m.weight)
            # m.bias.data.fill_(0.01)

    def reset_params(self):
        """(Re)set all parameters."""
        # self.sequential.apply(init_weights)
        layers = []
        for i in range(2):
            layers += [block1(self.input_units, self.output_units, self.hidden_units)]
        self.network = nn.Sequential(*layers)
        pass

    def forward(self, X):  # pylint: disable=arguments-differ
        out = self.network(X)
        out = self.linear(out)
        return out
