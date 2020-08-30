# explore the algorithm wrapped by RFE

import sys
sys.path.insert(0, '/home/fuzzy/work/DeepFlarePred/')
sys.path.insert(0, '/home/dewaldus/projects/DeepFlarePred/')
from data_loader import data_loader
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
# explore the algorithm wrapped by RFE
from sklearn.tree import DecisionTreeClassifier

n_features = 40
start_feature = 5
mask_value = 0
drop_path = os.path.expanduser(
    '~/Dropbox/_Meesters/figures/features_inspect/')
filepath = '../Data/Liu/M5_only/'

feature_list = None

X_train_data, y_train_data = data_loader.load_data(
        datafile=filepath + 'normalized_training.csv',
        flare_label='M5', series_len=1,
        start_feature=start_feature, n_features=n_features,
        mask_value=mask_value, feature_list=feature_list)
X_valid_data, y_valid_data = data_loader.load_data(
        datafile=filepath + 'normalized_validation.csv',
        flare_label='M5', series_len=1,
        start_feature=start_feature, n_features=n_features,
        mask_value=mask_value, feature_list=feature_list)
X_test_data, y_test_data = data_loader.load_data(
        datafile=filepath + 'normalized_testing.csv',
        flare_label='M5', series_len=1,
        start_feature=start_feature, n_features=n_features,
        mask_value=mask_value, feature_list=feature_list)
X_train_data = np.reshape(X_train_data, (len(X_train_data), n_features))
X_valid_data = np.reshape(X_valid_data, (len(X_valid_data), n_features))
X_test_data = np.reshape(X_test_data, (len(X_test_data), n_features))
# combine train and validation
X = X_train_data
y = y_train_data
# X = np.concatenate((X_train_data, X_valid_data))
# y = np.concatenate((y_train_data, y_valid_data))

y_train_tr = data_loader.label_transform(y_train_data)
y_valid_tr = data_loader.label_transform(y_valid_data)
y_test_tr = data_loader.label_transform(y_test_data)
y = data_loader.label_transform(y)

feature_name = pd.DataFrame(data_loader.get_feature_names(
            filepath + 'normalized_training.csv'))[5:]


#get the dataset
def get_dataset():
    X, y = make_classification(n_samples=1000, n_features=10, n_informative=5,
                               n_redundant=5, random_state=1)
    return X, y


# get a list of models to evaluate
def get_models():
    models = dict()
    # lr
    model = LogisticRegression()
    models['lr'] = Pipeline(steps=[('m', model)])
    # perceptron
    model = Perceptron()
    models['per'] = Pipeline(steps=[('m', model)])
    # cart
    model = DecisionTreeClassifier()
    models['cart'] = Pipeline(steps=[('m', model)])
    # rf
    model = RandomForestClassifier()
    models['rf'] = Pipeline(steps=[('m', model)])
    # gbm
    model = GradientBoostingClassifier()
    models['gbm'] = Pipeline(steps=[('m', model)])
    return models


# evaluate a give model using cross-validation
def evaluate_model(model):
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=1)
    scores = cross_val_score(model, X, y,
                             scoring=make_scorer(balanced_accuracy_score,
                                                 **{'adjusted': True}), cv=cv,
                             n_jobs=-1)
    return scores


def train_test_model(model, X_test, y_test):
    model.fit(X,y)
    test_tss = balanced_accuracy_score(y_true=y_test,
                                       y_pred=model.predict(X_test),
                                       adjusted=True)
    return test_tss

# define dataset
# X, y = get_dataset()
# get the models to evaluate
models = get_models()
# evaluate the models and store results
results, names = list(), list()
for name, model in models.items():
    scores = evaluate_model(model)
    results.append(scores)
    names.append(name)
    print('Train >%s %.3f (%.3f)' % (name, np.mean(scores), np.std(scores)))
    print(f"Val >{name} {train_test_model(model, X_valid_data, y_valid_tr)}")
    print(f"Test >{name} {train_test_model(model, X_test_data, y_test_tr)}")
# plot model performance for comparison
plt.boxplot(results, labels=names, showmeans=True)
plt.show()