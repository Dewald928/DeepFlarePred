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

    def reset_params(self):
        """(Re)set all parameters."""
        units = [self.input_units]
        units += [self.hidden_units] * self.num_hidden
        units += [self.output_units]

        sequence = []
        for u0, u1 in zip(units, units[1:]):
            sequence.append(nn.Linear(u0, u1))
            sequence.append(self.nonlin)
            sequence.append(nn.Dropout(self.dropout))

        sequence = sequence[:-2]
        if self.output_nonlin:
            sequence.append(self.output_nonlin)

        self.sequential = nn.Sequential(*sequence)

    def forward(self, X):  # pylint: disable=arguments-differ
        X = self.sequential(X)
        if self.squeeze_output:
            X = X.squeeze(-1)
        return F.softmax(X, dim=1)