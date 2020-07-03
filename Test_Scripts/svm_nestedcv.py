import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.metrics import balanced_accuracy_score, classification_report
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold, \
    StratifiedKFold, KFold
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
# explore the algorithm wrapped by RFE
from sklearn.svm import LinearSVC, l1_min_c
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from tabulate import tabulate

from data_loader import data_loader
from utils import confusion_matrix_plot

# Data
n_features = 32
start_feature = 5
mask_value = 0
drop_path = os.path.expanduser(
    '~/Dropbox/_Meesters/figures/features_inspect/')
filepath = '../Data/Liu/' + 'M5' + '/'

# todo removed redundant features
listofuncorrfeatures = ['TOTUSJH', 'ABSNJZH', 'TOTUSJZ', 'TOTBSQ', 'USFLUX',
                        'Cdec', 'Chis', 'Edec', 'Mhis', 'Xmax1d', 'Mdec',
                        'AREA_ACR', 'MEANPOT', 'Mhis1d', 'SHRGT45', 'TOTFX',
                        'MEANSHR', 'MEANGBT', 'TOTFZ', 'TOTFY', 'logEdec',
                        'EPSZ', 'MEANGBH', 'MEANGBZ', 'Xhis1d', 'Xdec', 'Xhis',
                        'EPSX', 'EPSY', 'Bhis', 'Bdec', 'Bhis1d']
feature_list = listofuncorrfeatures

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
X = np.concatenate((X_train_data, X_valid_data))
y = np.concatenate((y_train_data, y_valid_data))

y_train_tr = data_loader.label_transform(y_train_data)
y_valid_tr = data_loader.label_transform(y_valid_data)
y_test_tr = data_loader.label_transform(y_test_data)
y = data_loader.label_transform(y)

feature_name = pd.DataFrame(data_loader.get_feature_names(
            filepath + 'normalized_training.csv'))[5:]

# example data
# X, y = make_classification(n_samples=10000, n_features=10, n_redundant=5,
#                            n_informative=5, n_clusters_per_class=1,
#                            weights=[0.1, 0.9], class_sep=1)
# X, X_test_data, y, y_test_tr = train_test_split(X,y, test_size=0.3)


'''
SVM Nested CV
'''
seed = 4
params = {'C': [0.001]}  # choose C values here
clf = LinearSVC(penalty="l1", dual=False, verbose=0, max_iter=10000,
                class_weight='balanced')

inner_cv = KFold(n_splits=2, shuffle=True, random_state=seed)
outer_cv = KFold(n_splits=5, shuffle=True, random_state=seed)

gcv = GridSearchCV(estimator=clf, param_grid=params,
                   scoring=make_scorer(balanced_accuracy_score,
                                       **{'adjusted': True}),
                   n_jobs=-1, cv=inner_cv, refit=True,
                   return_train_score=True, verbose=1)

nested_score = cross_val_score(gcv, X=X, y=y,
                               cv=outer_cv, n_jobs=-1,
                               scoring=make_scorer(
                                   balanced_accuracy_score,
                                   **{'adjusted': True}))
print('%s | outer TSS %.4f +/- %.4f' % (
    'SVM', nested_score.mean(), nested_score.std()))

# Fitting a model to the whole training set
# using the "best" algorithm
# best_algo = gridcvs['SVM']
gcv.fit(X, y)

# summarize results
print("Best: %f using %s" % (
gcv.best_score_, gcv.best_params_))
means = gcv.cv_results_['mean_test_score']
stds = gcv.cv_results_['std_test_score']
params = gcv.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

train_tss = balanced_accuracy_score(y_true=y,
                           y_pred=gcv.predict(X), adjusted=True)
test_tss = balanced_accuracy_score(y_true=y_test_tr,
                          y_pred=gcv.predict(X_test_data), adjusted=True)

print('Training TSS: %.4f' % (train_tss))
print('Test TSS: %.4f' % (test_tss))


# save to csv
nested_score_df = pd.Series(nested_score, name='best_outer_score')
df_csv = pd.concat([pd.DataFrame(gcv.cv_results_), nested_score_df], axis=1)
df_csv.to_csv('../saved/scores/nestedcv_svm_{}.csv'.format(seed))

# calibration
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

cclf = CalibratedClassifierCV(base_estimator=LinearSVC(penalty='l2',
                                                       dual=False, C=0.001), cv=5)
cclf.fit(X, y)
res = cclf.predict_proba(X_test_data)[:, 1]

# or use
y_pred = gcv.decision_function(X_test_data)