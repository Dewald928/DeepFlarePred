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

# padding and masking
np.random.seed(1)
X = np.random.random((4, 6)).round(1) * 2 + 3
X = torch.from_numpy(X)
X_len = torch.LongTensor([4, 1, 6, 3])
maxlen = X.size(1)
mask = torch.arange(maxlen)[None, :] < X_len[:, None]

X[~mask] = float('-inf')
out = torch.softmax(X, dim=1)


# pack sequence
# x_padded = pad_sequence(x_seq, batch_first=True, padding_value=0)
packed = torch.nn.utils.rnn.pack_padded_sequence(out, X_len, batch_first=True,
                                                 enforce_sorted=False)
