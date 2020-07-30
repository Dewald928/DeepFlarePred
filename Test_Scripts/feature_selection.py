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
from sklearn.model_selection import RepeatedStratifiedKFold, KFold, RepeatedKFold
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
# explore the algorithm wrapped by RFE
from sklearn.svm import LinearSVC, l1_min_c
from sklearn.tree import DecisionTreeClassifier
from tabulate import tabulate
from joblib import dump, load

import sys
sys.path.insert(0, '/home/fuzzy/work/DeepFlarePred/')
sys.path.insert(0, '/home/dewaldus/projects/DeepFlarePred/')
from data_loader import data_loader
from utils import confusion_matrix_plot


class GridSearchWithCoef(GridSearchCV):
    @property
    def coef_(self):
        return self.best_estimator_.coef_

# Data
n_features = 40
start_feature = 5
mask_value = 0
drop_path = os.path.expanduser('~/Dropbox/_Meesters/figures/features_inspect/')
# drop_path = os.path.expanduser(
#     '~/projects/DeepFlarePred/saved/features_inspect/')
filepath = './Data/Liu/' + 'M5' + '/'

# todo removed redundant features
listofuncorrfeatures = ['TOTUSJH', 'SAVNCPP', 'ABSNJZH', 'TOTPOT', 'AREA_ACR',
                        'Cdec', 'Chis', 'Edec', 'Mhis', 'Xmax1d', 'Mdec',
                        'MEANPOT', 'R_VALUE', 'Mhis1d', 'MEANGAM', 'TOTFX',
                        'MEANJZH', 'MEANGBZ', 'TOTFZ', 'TOTFY', 'logEdec',
                        'EPSZ', 'MEANGBH', 'MEANJZD', 'Xhis1d', 'Xdec', 'Xhis',
                        'EPSX', 'EPSY', 'Bhis', 'Bdec', 'Bhis1d']  # 32
# feature_list = ['Bdec', 'Cdec', 'Chis1d', 'Edec', 'MEANGBZ', 'Mdec',
#                 'Mhis1d', 'R_VALUE', 'Xdec', 'Xhis1d', 'Xmax1d']
feature_list = None

X_train_data, y_train_data = data_loader.load_data(
    datafile=filepath + 'normalized_training.csv', flare_label='M5',
    series_len=1, start_feature=start_feature, n_features=n_features,
    mask_value=mask_value, feature_list=feature_list)
X_valid_data, y_valid_data = data_loader.load_data(
    datafile=filepath + 'normalized_validation.csv', flare_label='M5',
    series_len=1, start_feature=start_feature, n_features=n_features,
    mask_value=mask_value, feature_list=feature_list)
X_test_data, y_test_data = data_loader.load_data(
    datafile=filepath + 'normalized_testing.csv', flare_label='M5',
    series_len=1, start_feature=start_feature, n_features=n_features,
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

feature_name = pd.DataFrame(
    data_loader.get_feature_names(filepath + 'normalized_training.csv'))[
               5:].reset_index(drop=True)
feature_names = data_loader.get_feature_names(
    filepath + 'normalized_training.csv')

# X, y = make_classification(n_samples=10000, n_features=10, n_redundant=5,
#                            n_informative=5, n_clusters_per_class=1,
#                            weights=[0.1, 0.9], class_sep=1)
# X, X_test_data, y, y_test_tr = train_test_split(X,y, test_size=0.3)

'''
Univariate Feature selection
'''
f_df = pd.DataFrame()
mi_df = pd.DataFrame()
score_functions = {'F-score': f_classif,
                   'Mutual-Information': mutual_info_classif}
for scorer, score_func in score_functions.items():
    print(scorer, score_func)
    selector = SelectKBest(score_func, k=40)
    selected_features = selector.fit_transform(X, y)
    plt.bar(
        feature_names[
        5:], selector.scores_)
    plt.title('{} vs. Features'.format(scorer))
    plt.xlabel('Features')
    plt.xticks(rotation=90)
    plt.ylabel('Importance')
    plt.tight_layout()
    plt.savefig(drop_path + "{}.png".format(scorer))
    plt.show()
    plt.close('all')
    f_score_indexes = (-selector.scores_).argsort()[:40]

    sort_df = pd.concat([pd.DataFrame(
        feature_names[
        5:], columns=['Features']), pd.DataFrame(selector.scores_,
        columns=['Importance'])], axis=1).sort_values(by='Importance',
        ascending=False).reset_index()
    f_df = sort_df if scorer == 'F-score' else f_df
    mi_df = sort_df if scorer == 'Mutual-Information' else mi_df
    sns.barplot(x='Features', y='Importance', data=sort_df,
                order=sort_df['Features'])
    plt.xticks(rotation=90)
    plt.title('Sorted {} vs. Features'.format(scorer))
    plt.tight_layout()
    plt.savefig(drop_path + "{}_sorted.png".format(scorer))
    plt.show()
    plt.close('all')

'''
Linear SVC optimization
'''
params = {'C': [0.000001,0.00001,0.0001,0.001,0.01,0.1,1,10,100]}  # choose C values here
clf = LinearSVC(penalty="l2", dual=False, verbose=0, max_iter=10000,
                class_weight='balanced')
# cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=1)
grid = GridSearchCV(clf, param_grid=params, cv=5, return_train_score=True,
                    scoring=make_scorer(balanced_accuracy_score,
                                        **{'adjusted': True}), n_jobs=-1,
                    verbose=1)
grid.fit(X, y)

print("Cross-validation grid scores on development set:")
print('')
tmeans = grid.cv_results_['mean_train_score']
tstds = grid.cv_results_['std_train_score']
means = grid.cv_results_['mean_test_score']
stds = grid.cv_results_['std_test_score']
print('| Parameter | Training TSS ($\mu$ ± '
      '$\sigma$) | Validation TSS ($\mu$ ± '
      '$\sigma$) |')
print('|---|---|---|')
for tmean, tstd, mean, std, params in zip(tmeans, tstds, means, stds,
                                          grid.cv_results_['params']):
    print(f"| {params} | {tmean:0.4f} ± {tstd:0.4f} | {mean:0.4f} ±"
          f" {std:0.4f} |")

print("Best parameters set found on development set:")
print(grid.best_params_)

print("The model is trained on the full development set.")
print("The scores are computed on the full evaluation set.")
print('')
train_tss = balanced_accuracy_score(y_true=y,
                           y_pred=grid.predict(X), adjusted=True)
test_tss = balanced_accuracy_score(y_true=y_test_tr,
                          y_pred=grid.predict(X_test_data), adjusted=True)
print('| Dataset | TSS |')
print("|---|---|")
print(f"| Training + Validation | {train_tss:.4f} |")
print(f"| Test | {test_tss:.4f} |")

# plot weights
svm_weights = np.abs(grid.best_estimator_.coef_).sum(axis=0)
svm_weights /= svm_weights.sum()
X_indices = np.arange(X.shape[-1])
plt.bar(X_indices, svm_weights, width=.2, label=f"C:{grid.best_estimator_.C}")

plt.title("SVM Coefficients")
plt.xlabel('Features')
plt.xticks(range(0, len(feature_names[5:])), feature_names[5:], rotation=90)
plt.yticks(())
plt.axis('tight')
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig(drop_path + "SVM_coef.png")
plt.show()

'''
Crossval from feature selection
'''
cs = l1_min_c(X, y)  # minimum C-value
clf = Pipeline([('anova', SelectKBest(f_classif)), ('svc', LinearSVC(
    C=0.001, penalty="l2", dual=False, verbose=0, max_iter=10000,
    class_weight='balanced'))])

# Plot the cross-validation score as a function of number of features
score_means = list()
score_stds = list()
n_features_list = np.arange(40, 0, -1)

for k_feature in n_features_list:
    clf.set_params(anova__k=k_feature)
    # cv = RepeatedStratifiedKFold(n_splits=5)
    this_scores = cross_val_score(clf, X, y, cv=5,
                                  scoring=make_scorer(balanced_accuracy_score,
                                                      **{'adjusted': True}),
                                  n_jobs=-1, verbose=1)
    score_means.append(this_scores.mean())
    score_stds.append(this_scores.std())

plt.errorbar(n_features_list, score_means, score_stds)
plt.ylim(0,1)
plt.title(
    'Performance of the SVM varying the number of features selected')
plt.xticks(np.linspace(n_features, 0, 11, endpoint=True))
plt.xlabel('Number of Features')
plt.ylabel('TSS')
plt.tight_layout()
plt.savefig(drop_path + "SVM_CV_F.png")
plt.show()

'''
Recursive Feature Elimination
'''
clf = LinearSVC(penalty="l2", dual=False, verbose=0, max_iter=10000,
                class_weight='balanced', random_state=1, C=0.001)

# Get feature ranking
try:
    # reload model
    rfecv = load(f'../saved/models/SVM/rfecv_{clf.C:.0e}.joblib')
except:
    rfecv = RFECV(clf, cv=5, scoring=make_scorer(balanced_accuracy_score,
                                                 **{'adjusted': True}),
                  n_jobs=-1, verbose=1)

# grid = GridSearchWithCoef(rfecv, param_grid={'estimator__C':[0.0001,0.001,0.01,0.1,1,10]})
# grid.fit(X,y)
rfecv.fit(X, y)

# plot scores
plt.figure(figsize=(16, 9))
plt.title('Recursive Feature Elimination with Cross-Validation')
plt.xlabel('Number of features selected')
plt.ylabel('TSS')
plt.ylim(0,1)
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.tight_layout()
plt.savefig(drop_path+'RFECV_avg.png')
plt.show()

print("Features sorted by their rank:")
rfe_df = pd.DataFrame(
    sorted(zip(map(lambda x: round(x, 4), rfecv.ranking_), feature_names[5:])))
print(tabulate(rfe_df.T, headers="keys", tablefmt="github", showindex=False))

# RFE evaluation
print("The model is trained on the full development set.")
print("The scores are computed on the full evaluation set.")
print('')
trainval_tss = balanced_accuracy_score(y_true=y,
                           y_pred=rfecv.predict(X), adjusted=True)
test_tss = balanced_accuracy_score(y_true=y_test_tr,
                          y_pred=rfecv.predict(X_test_data), adjusted=True)
print('| Dataset | TSS |')
print("|---|---|")
print(f"| Training + Validation | {trainval_tss:.4f} |")
print(f"| Test | {test_tss:.4f} |")
confusion_matrix_plot.plot_confusion_matrix_from_data(y_test_tr, rfecv.predict(X_test_data),
                                                      ['Negative', 'Positive'])

# save model
dump(rfecv, f'../saved/models/SVM/rfecv_{clf.C:.0e}.joblib')

# get a list of models to evaluate
def get_models():
    models = dict()
    for i in range(1, 41):
        rfe = RFE(estimator=clf, n_features_to_select=i)
        model = clf
        models[str(i)] = Pipeline(steps=[('s', rfe), ('m', model)])
    return models


# evaluate a give model using cross-validation
def evaluate_model(model):
    # cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=1)
    scores = cross_val_score(model, X, y,
                             scoring=make_scorer(balanced_accuracy_score,
                                                 **{'adjusted': True}), cv=5,
                             n_jobs=-1, error_score='raise')
    return scores


# get the models to evaluate
models = get_models()
# evaluate the models and store results
results, names = list(), list()
for name, model in models.items():
    scores = evaluate_model(model)
    results.append(scores)
    names.append(name)
    print('>%s %.3f (%.3f)' % (name, np.mean(scores), np.std(scores)))
# plot model performance for comparison
fig = plt.figure(figsize=(10, 8))
# plt.boxplot(results, labels=names, showmeans=True, showfliers=False)
plt.errorbar(np.arange(n_features), np.mean(results, axis=1),
             np.std(results, axis=1))
plt.title('Recursive Feature Elimination with Cross-Validation')
plt.xlabel('Number of features selected')
plt.ylabel('TSS')
plt.ylim(0,1)
plt.tight_layout()
plt.savefig(drop_path + 'RFECV_boxplot.png')
plt.show()
# rfe_results = pd.concat([pd.DataFrame(names), pd.DataFrame(results)], axis=1)
# rfe_results.to_csv('rfe_results.csv')

# rfe_results = pd.read_csv('rfe_results.csv')
# results = rfe_results.iloc[:, 2:]
# names = rfe_results.index.values


'''
RFE Multiple algorithms test
'''
# get the dataset
# def get_dataset():
#     X, y = make_classification(n_samples=1000, n_features=10, n_informative=5,
#                                n_redundant=5, random_state=1)
#     return X, y
#
#
# # get a list of models to evaluate
# def get_models():
#     models = dict()
#     # lr
#     rfe = RFE(estimator=LogisticRegression(), n_features_to_select=5)
#     model = DecisionTreeClassifier()
#     models['lr'] = Pipeline(steps=[('s', rfe), ('m', model)])
#     # perceptron
#     rfe = RFE(estimator=Perceptron(), n_features_to_select=5)
#     model = DecisionTreeClassifier()
#     models['per'] = Pipeline(steps=[('s', rfe), ('m', model)])
#     # cart
#     rfe = RFE(estimator=DecisionTreeClassifier(), n_features_to_select=5)
#     model = DecisionTreeClassifier()
#     models['cart'] = Pipeline(steps=[('s', rfe), ('m', model)])
#     # rf
#     rfe = RFE(estimator=RandomForestClassifier(), n_features_to_select=5)
#     model = DecisionTreeClassifier()
#     models['rf'] = Pipeline(steps=[('s', rfe), ('m', model)])
#     # gbm
#     rfe = RFE(estimator=GradientBoostingClassifier(), n_features_to_select=5)
#     model = DecisionTreeClassifier()
#     models['gbm'] = Pipeline(steps=[('s', rfe), ('m', model)])
#     return models
#
#
# # evaluate a give model using cross-validation
# def evaluate_model(model):
#     cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=1)
#     scores = cross_val_score(model, X, y,
#                              scoring=make_scorer(balanced_accuracy_score,
#                                                  **{'adjusted': True}), cv=cv,
#                              n_jobs=-1)
#     return scores
#
#
# # define dataset
# # X, y = get_dataset()
# # get the models to evaluate
# models = get_models()
# # evaluate the models and store results
# results, names = list(), list()
# for name, model in models.items():
#     scores = evaluate_model(model)
#     results.append(scores)
#     names.append(name)
#     print('>%s %.3f (%.3f)' % (name, np.mean(scores), np.std(scores)))
# # plot model performance for comparison
# plt.boxplot(results, labels=names, showmeans=True)
# plt.show()

'''
Comparison Plot
'''
f_df_c = f_df.loc[:, ['Features']]
f_df_c = pd.DataFrame(f_df_c.index.values, index=f_df_c['Features'])

mi_df_c = mi_df.loc[:, ['Features']]
mi_df_c = pd.DataFrame(mi_df_c.index.values, index=mi_df_c['Features'])

feature_name.columns = ['Features']
liu_df_c = pd.DataFrame(feature_name.index.values, index=feature_name[
    'Features'])

rfe_df.columns = ['0', 'Features']
rfe_df_c = pd.DataFrame(rfe_df['0'].values, index=rfe_df['Features'])

df_comp = pd.concat([liu_df_c, f_df_c, mi_df_c, rfe_df_c], axis=1, sort=False)
df_comp.columns = ['Liu', 'F-Score', 'MI', 'RFE']
plt.figure(figsize=(20, 16))
ax = plt.subplot(111)
sns.heatmap(df_comp, ax=ax, annot=True, fmt='.1f')
plt.tight_layout()
plt.savefig(drop_path + "feature_selection_comparison.png")
plt.show()

sns.clustermap(df_comp, figsize=(20, 16))
plt.savefig(drop_path + 'feature_selection_comparison_clustermap.png')
plt.show()
