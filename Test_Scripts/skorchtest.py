from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from skorch import NeuralNetClassifier
from skorch.dataset import Dataset
from skorch.helper import predefined_split
from skorch.toy import make_classifier
import numpy as np

X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0,
                           n_informative=1, n_clusters_per_class=1,
                           weights=[0.1, 0.9], class_sep=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,
                                                    random_state=1)
X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)
y_train = y_train.astype(np.int64)
y_test = y_test.astype(np.int64)

valid_ds = Dataset(X_test, y_test)

model = make_classifier(**{"input_units": 2})

net = NeuralNetClassifier(
    model,
    max_epochs=10,
    lr=0.1,
    train_split=predefined_split(valid_ds),
)

net.fit(X_train, y_train)
y_proba = net.predict_proba(X_train)








