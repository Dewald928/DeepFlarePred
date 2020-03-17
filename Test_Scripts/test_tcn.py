import torch
from torch import nn
from model import tcn
import numpy as np


a = torch.tensor(
    np.array([[[0, 0, 1, 1], [0,0,1,1], [-1,0,0,0]]]),
    dtype=torch.float32)
# a = torch.randn(2, 3, 2)  # timesteps, features, sequence length
b = a.numpy()
m = nn.Conv1d(3, 3, 2, bias=False)
m.weight.data = torch.tensor([[[0, 0],
                             [1, 0],
                             [0, 0]],
                            [[2, 2],
                             [2, 2],
                             [2, 2]],
                            [[0, 0],
                             [0, 0],
                             [0, 0]]],
                             dtype=torch.float32)
kernel = m.weight.data.numpy()
n = nn.ReLU(m)
out = m(a)
out_np = out.detach().numpy()
