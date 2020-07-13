import torch
from torch import nn
from model import tcn
import numpy as np
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils import weight_norm

from model import tcn


def create_sequences(data, seq_length):
    xs = []
    for i in range(len(data)-seq_length-1):
        x = data[i:(i+seq_length)]
        xs.append(x)
    return np.array(xs)


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
relu1 = nn.ReLU()
linear = nn.Linear(3, 2)


a = torch.tensor(np.array([[[1, 0, 1, 1],
                            [0, 0, 1, 0],
                            [-1, 0, 1, 1]]]), dtype=torch.float32)
# a = torch.randn(2, 3, 2)  # timesteps, features, sequence length
b = a.numpy()
m = nn.Conv1d(3, 3, 2, bias=False, padding=1)
m.weight.data = torch.tensor([[[1, 0],
                             [0, 0],
                             [0, 1]],
                            [[2, 2],
                             [2, 2],
                             [2, 2]],
                            [[0, 0],
                             [0, 0],
                             [0, 1]]],
                             dtype=torch.float32)
kernel = m.weight.data.numpy()
# n = nn.ReLU(m)
out = m(a[:, :, :-1])
out = relu1(out)
out = linear(out[:, :, -1])
out_np = out.detach().numpy()


'''1dconv testing'''
# disable biases in conv and linear to make it easier
model = TCN(2, 2, [1],
                    kernel_size=2, dropout=0.0)
model = tcn.Simple1DConv(2, 1, kernel_size=2,
                         dropout=0)
model.conv1.weight_v.data = torch.tensor([[[1, 0],
                                           [0, 0]]],
                                         dtype=torch.float32)
model.conv1.weight_g.data = torch.tensor(1)
model.linear.weight.data = torch.tensor([[1],
        [1]], dtype=torch.float32)
# test samples
X = np.array([[1,0,0,1,0,0,1,0,0],
     [0,0,0,0,0,0,0,0,0]]).T
x_seq = create_sequences(X, 3)
X_tensor = torch.from_numpy(x_seq).float()
X_tensor = X_tensor.permute(0,2,1)

out = model(X_tensor)
out_np = out.detach().numpy()

print('finished')


