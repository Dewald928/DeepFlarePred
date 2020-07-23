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

import sys
sys.path.insert(0, '/home/fuzzy/work/DeepFlarePred/')
sys.path.insert(0, '/home/dewaldus/projects/DeepFlarePred/')
from data_loader import data_loader
from utils import confusion_matrix_plot


# Data
n_features = 40
start_feature = 5
mask_value = 0
drop_path = os.path.expanduser(
    '~/Dropbox/_Meesters/figures/features_inspect/')
filepath = '../Data/Liu/' + 'M5' + '/'

# todo removed redundant features
listofuncorrfeatures = ['TOTUSJH', 'SAVNCPP', 'ABSNJZH', 'TOTPOT', 'AREA_ACR',
                        'Cdec', 'Chis', 'Edec', 'Mhis', 'Xmax1d', 'Mdec',
                        'MEANPOT', 'R_VALUE', 'Mhis1d', 'MEANGAM', 'TOTFX',
                        'MEANJZH', 'MEANGBZ', 'TOTFZ', 'TOTFY', 'logEdec',
                        'EPSZ', 'MEANGBH', 'MEANJZD', 'Xhis1d', 'Xdec', 'Xhis',
                        'EPSX', 'EPSY', 'Bhis', 'Bdec', 'Bhis1d']  # 32
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
seed = 10
params = {'C': [0.000001,0.00001,0.0001,0.001,0.01,0.1,1,10,100]}  # choose C
clf = LinearSVC(penalty="l2", dual=False, verbose=0, max_iter=10000,
                class_weight='balanced')

inner_cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

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

# Fitting a model to the whole training set
# using the "best" algorithm
# best_algo = gridcvs['SVM']
gcv.fit(X, y)

# summarize results
tmeans = gcv.cv_results_['mean_train_score']
tstds = gcv.cv_results_['std_train_score']
means = gcv.cv_results_['mean_test_score']
stds = gcv.cv_results_['std_test_score']
print("Nested cross-validation grid scores on development set:")
print('')
print('| Parameter | Training TSS ($\mu$ ± '
      '$\sigma$) | Inner Fold Validation TSS ($\mu$ ± '
      '$\sigma$) |')
print('|---|---|---|')
for tmean, tstd, mean, std, params in zip(tmeans, tstds, means, stds,
                                          gcv.cv_results_['params']):
    print(f"| {params} | {tmean:0.4f} ± {tstd:0.4f} | {mean:0.4f} ±"
          f" {std:0.4f} |")

train_tss = balanced_accuracy_score(y_true=y,
                           y_pred=gcv.predict(X), adjusted=True)
test_tss = balanced_accuracy_score(y_true=y_test_tr,
                          y_pred=gcv.predict(X_test_data), adjusted=True)
print(f"Best parameter: {gcv.best_params_}  ")
print(f"Outer-fold TSS: {nested_score.mean():.4f} ± {nested_score.std():.4f}  ")

print("The model is trained on the full development set.  ")
print("The scores are computed on the full evaluation set.  ")
print('')
print("| Dataset | TSS |")
print("|---|---|")
print('| Training + Validation | %.4f |' % (train_tss))
print('| Test | %.4f |' % (test_tss))


# save to csv
nested_score_df = pd.Series(nested_score, name='best_outer_score')
df_csv = pd.concat([pd.DataFrame(gcv.cv_results_), nested_score_df], axis=1)
df_csv.to_csv('../saved/scores/nestedcv_svm_{}_strat.csv'.format(seed))

# calibration
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

cclf = CalibratedClassifierCV(base_estimator=LinearSVC(penalty='l2',
                                                       dual=False, C=0.001), cv=5)
cclf.fit(X, y)
res = cclf.predict_proba(X_test_data)[:, 1]

# or use
y_pred = gcv.decision_function(X_test_data)



