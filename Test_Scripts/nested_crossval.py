from sklearn.datasets import load_iris
from sklearn.utils import class_weight
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold, \
    StratifiedKFold, train_test_split
from sklearn.datasets import make_classification
import numpy as np
from skorch import NeuralNetClassifier
from skorch.dataset import Dataset
from skorch.toy import make_classifier
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
# from mlxtend.data import mnist_data


# Load the dataset
X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0,
                           n_informative=1, n_clusters_per_class=1,
                           weights=[0.1, 0.9], class_sep=1)
X = X.astype(np.float32)
ds = Dataset(X, y)
y = np.array([y for _, y in iter(ds)])
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    train_size=0.8,
                                                    random_state=1,
                                                    stratify=y)

class_weights = class_weight.compute_class_weight('balanced',
                                                  np.unique(y),
                                                  y)

# Set up possible values of parameters to optimize over
p_grid = {"module__num_hidden": [1],
          "module__hidden_units": [40],
          "lr": [1, 0.1]}

# Model
model = make_classifier(**{"input_units": 2})
# noinspection PyArgumentList
net = NeuralNetClassifier(model, lr=1,
                          batch_size=64,
                          max_epochs=200,
                          criterion=nn.CrossEntropyLoss,
                          criterion__weight=torch.FloatTensor(
                              class_weights),
                          # callbacks=[bacc, train_acc, roc_auc,
                          #            EarlyStopping(monitor='train_bacc',
                          #                          lower_is_better=False,
                          #                          patience=100),
                          #            checkpoint],
                          train_split=None,)

inner_cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=1)
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

gcv = GridSearchCV(estimator=net, param_grid=p_grid, scoring='accuracy',
                   n_jobs=1, cv=inner_cv, refit=True)

nested_score = cross_val_score(gcv, X=X_train, y=y_train, cv=outer_cv,
                               n_jobs=1)
print('%s | outer ACC %.2f%% +/- %.2f' % (
    'MLP', nested_score.mean() * 100, nested_score.std() * 100))

# Fitting a model to the whole training set
# using the "best" algorithm
best_algo = gcv

best_algo.fit(X_train, y_train)
train_acc = accuracy_score(y_true=y_train, y_pred=best_algo.predict(X_train))
test_acc = accuracy_score(y_true=y_test, y_pred=best_algo.predict(X_test))

print('Accuracy %.2f%% (average over CV test folds)' %
      (100 * best_algo.best_score_))
print('Best Parameters: %s' % best_algo.best_params_)
print('Training Accuracy: %.2f%%' % (100 * train_acc))
print('Test Accuracy: %.2f%%' % (100 * test_acc))
