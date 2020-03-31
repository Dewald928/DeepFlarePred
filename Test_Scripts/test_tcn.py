import torch
from torch import nn
from model import tcn
import numpy as np
from torch import nn
from torch.nn.utils.rnn import pad_sequence


a = torch.tensor(np.array([[[0, 0, 1, 1],
                            [0, 0, 1, 1],
                            [-1, 0, 0, 0]]]), dtype=torch.float32)
# a = torch.randn(2, 3, 2)  # timesteps, features, sequence length
b = a.numpy()
m = nn.Conv1d(3, 3, 2, bias=False)
m.weight.data = torch.tensor([[[1, 0],
                             [0, 1],
                             [1, 1]],
                            [[2, 2],
                             [2, 2],
                             [2, 2]],
                            [[0, 0],
                             [0, 0],
                             [0, 0]]],
                             dtype=torch.float32)
kernel = m.weight.data.numpy()
# n = nn.ReLU(m)
out = m(a)
out_np = out.detach().numpy()

# cross val
from sklearn.model_selection import TimeSeriesSplit
X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4]])
y = np.array([1, 2, 3, 4, 5, 6])
tscv = TimeSeriesSplit(n_splits=8)
print(tscv)
for train, test in tscv.split(X_test_data):
    print("%s %s" % (train, test))
