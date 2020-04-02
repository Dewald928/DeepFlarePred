import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import make_scorer
from skorch import NeuralNetClassifier
from skorch.callbacks import EarlyStopping
from skorch.callbacks import EpochScoring
from skorch.dataset import Dataset
from skorch.callbacks import Checkpoint
from skorch.toy import make_classifier
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
import torch

np.random.seed(0)
torch.manual_seed(0)

X, y = make_classification(random_state=0)
X = X.astype(np.float32)
ds = Dataset(X, y)
y = np.array([y for _, y in iter(ds)])

acc = EpochScoring(scoring='accuracy',
                             lower_is_better=False, name='accuracy',
                             use_caching=True, on_train=True)

net = NeuralNetClassifier(
    make_classifier(),
    lr=1,
    max_epochs=100,
    callbacks=[EarlyStopping(monitor='valid_acc', lower_is_better=False)],
               # Checkpoint(monitor='accuracy_best')],
    # consider setting verbose=0
)

scores = cross_validate(net, X, y, scoring='accuracy', cv=5)
print(scores)

