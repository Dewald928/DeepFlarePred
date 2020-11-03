from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from skorch import NeuralNetClassifier
from skorch.dataset import Dataset
from skorch.helper import predefined_split
from skorch.toy import make_classifier
import numpy as np
import pandas as pd
import os

X, y = make_classification(n_samples=200000, n_features=2, n_redundant=0,
                           n_informative=1, n_clusters_per_class=1,
                           weights=[0.1, 0.9], class_sep=0.1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,
                                                    random_state=1)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train,
                                                      test_size=0.33,
                                                      random_state=1)
X_train = X_train.astype(np.float32)
X_valid = X_valid.astype(np.float32)
X_test = X_test.astype(np.float32)
y_train = y_train.astype(np.int64)
y_valid = y_valid.astype(np.int64)
y_test = y_test.astype(np.int64)

t_train = pd.concat([pd.DataFrame(y_train), pd.DataFrame(X_train)], axis=1)
t_val = pd.concat([pd.DataFrame(y_valid), pd.DataFrame(X_valid)], axis=1)
t_test = pd.concat([pd.DataFrame(y_test), pd.DataFrame(X_test)], axis=1)

# save to synth
new_path = '../Data/Synth/'
if not os.path.exists(new_path):
    os.makedirs(new_path)
t_train.to_csv(new_path + 'normalized_training.csv', index=False)
t_val.to_csv(new_path + 'normalized_validation.csv', index=False)
t_test.to_csv(new_path + 'normalized_testing.csv', index=False)

valid_ds = Dataset(X_test, y_test)

model = make_classifier(**{"input_units": 2})

net = NeuralNetClassifier(model, max_epochs=10, lr=0.1,
    train_split=predefined_split(valid_ds), )

net.fit(X_train, y_train)
y_proba = net.predict_proba(X_train)
