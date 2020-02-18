import torch
from torch import nn
from model import tcn
import numpy as np

X_train_data = X_valid_data = X_test_data = np.array([[1,2,3],[1,2,3],
                                                     [1,1,1],[1,1,1]])
X_train_data_tensor = torch.tensor(X_train_data).float()
X_train_data_tensor = X_train_data_tensor.view(len(X_train_data_tensor),
                                               3, 1)
x = X_train_data_tensor.numpy()
# y_train_data = y_valid_data = y_test_data = np.array([1,1,0,0])
a = torch.randn(2, 3, 2)  # timesteps, features, sequence length
# a = torch.from_numpy(X_test_data)
b = a.numpy()
m = nn.Conv1d(3, 3, 1)
n = nn.ReLU(m)
out = m(a)
out_np = out.detach().numpy()
print(out.size())
print(m)
