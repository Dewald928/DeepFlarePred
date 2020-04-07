import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import make_scorer, balanced_accuracy_score, confusion_matrix
from skorch import NeuralNetClassifier
from skorch.callbacks import EarlyStopping
from skorch.callbacks import EpochScoring
from skorch.dataset import Dataset
from skorch.callbacks import Checkpoint
from skorch.toy import make_classifier
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import torch
import torch.nn as nn
import math
import wandb
from main_TCN_Liu import TCN
# wandb.init(project="skorchcv", tags='Base')

np.random.seed(0)
torch.manual_seed(0)

# Data prepare
X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0,
                           n_informative=1, n_clusters_per_class=1,
                           weights=[0.1, 0.9], class_sep=1)
X = X.astype(np.float32)
ds = Dataset(X, y)
y = np.array([y for _, y in iter(ds)])
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
h = .02  # step size in the mesh
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
xx, yy = np.float32(xx), np.float32(yy)

class_weights = class_weight.compute_class_weight('balanced',
                                                  np.unique(y),
                                                  y)
# Callbacks
bacc = EpochScoring(
    scoring=make_scorer(balanced_accuracy_score, **{'adjusted': True}),
    lower_is_better=False, name='train_bacc', use_caching=True, on_train=True)
train_acc = EpochScoring(scoring='accuracy', lower_is_better=False,
                         name='train_acc', on_train=True)
roc_auc = EpochScoring(scoring='roc_auc', lower_is_better=False, on_train=True)
checkpoint = Checkpoint(monitor='train_bacc_best')

# Model
model = make_classifier(**{"input_units": 2})
# noinspection PyArgumentList
net = NeuralNetClassifier(model, lr=1,
                          max_epochs=400,
                          criterion=nn.CrossEntropyLoss,
                          criterion__weight=torch.FloatTensor(
                              class_weights),
                          callbacks=[bacc, train_acc, roc_auc,
                                     EarlyStopping(monitor='train_bacc',
                                                   lower_is_better=False,
                                                   patience=100),
                                     checkpoint],
                          # consider setting verbose=0
                          train_split=None,
                          warm_start=False)
# wandb.watch(model)

# Cross Validation
# scores = cross_validate(net, X, y, scoring=make_scorer(balanced_accuracy_score,
#                                                        **{'adjusted': True}),
#                         cv=5, return_train_score=True)
#
# print(scores)
# print("TSS: %0.2f (+/- %0.2f)" % (
#     scores['test_score'].mean(), scores['test_score'].std() / math.sqrt(len(
#         scores['test_score']))))
# net.initialize()  # This is important!
# net.load_params(f_params='some-file.pkl')
net.fit(X, y)
net.load_params(checkpoint=checkpoint)
y_pred = net.predict(X)
TSS = balanced_accuracy_score(y, y_pred, adjusted=True)
CM = confusion_matrix(y, y_pred)

# plot data
cm = plt.cm.RdBu
cm_bright = ListedColormap(['#FF0000', '#0000FF'])
ax = plt.subplot()


Z = net.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
Z = Z.reshape(xx.shape)
ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)
ax.scatter(X[:, 0], X[:, 1], c=y, s=30, edgecolor='k', cmap=cm_bright)
ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.text(xx.max() - 1.5, yy.min() + .2, ('TSS: %.2f \n[%d, %d] \n[%d, %d]' % (
    TSS, CM[0][0], CM[0][1], CM[1][0], CM[1][1])).lstrip('0'),
                size=15, horizontalalignment='left')
plt.tight_layout()
plt.show()

# saving
net.save_params(f_params='some-file.pkl')
