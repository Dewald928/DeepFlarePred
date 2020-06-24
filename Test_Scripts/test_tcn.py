import torch
from torch import nn
from model import tcn
import numpy as np
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils import weight_norm


# how to init weight
torch.manual_seed(3)
linear = weight_norm(nn.Conv1d(1,1,3), name='weight', dim=None)
torch.manual_seed(3)
linear.weight.data.normal_(0, 1)
torch.manual_seed(3)
linear.weight_v.data.normal_(0, 1)

torch.manual_seed(3)
linear2 = nn.Conv1d(1,1,3)
# linear2.weight.data.normal_(0, 1)

print(linear.weight)
print(linear2.weight)

# how to get causal conv
m = nn.Conv1d(1, 1, 3, stride=1, bias=False)
input = torch.randn(1, 1, 50)
output = m(input)


a = torch.tensor(np.array([[[0, 0, 1, 1],
                            [0, 0, 1, 1],
                            [-1, 0, 0, 0]]]), dtype=torch.float32)
# a = torch.randn(2, 3, 2)  # timesteps, features, sequence length
b = a.numpy()
m = nn.Conv1d(3, 3, 2, bias=False, padding=1)
m.weight.data = torch.tensor([[[0, 0],
                             [0, 0],
                             [0, 1]],
                            [[2, 2],
                             [2, 2],
                             [2, 2]],
                            [[0, 0],
                             [0, 0],
                             [0, 0]]],
                             dtype=torch.float32)
kernel = m.weight.data.numpy()
# n = nn.ReLU(m)
out = m(a[:, :, :-1])
out_np = out.detach().numpy()


