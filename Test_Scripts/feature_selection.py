from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif, mutual_info_classif, chi2
from sklearn.feature_selection import RFE, GenericUnivariateSelect
from sklearn.metrics import average_precision_score, make_scorer
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import class_weight
from sklearn.metrics import precision_recall_curve, balanced_accuracy_score,\
    classification_report
from sklearn.svm import LinearSVC, SVC, l1_min_c
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, GridSearchCV
import matplotlib.pyplot as plt
import numpy as np
from data_loader import data_loader
import pandas as pd
from tabulate import tabulate
import seaborn as sns

# Data
drop_path = os.path.expanduser(
    '~/Dropbox/_Meesters/figures/features_inspect/')
filepath = './Data/Liu/' + 'M5' + '/'
X_train_data, y_train_data = data_loader.load_data(
        datafile=filepath + 'normalized_training.csv',
        flare_label=cfg.flare_label, series_len=1,
        start_feature=start_feature, n_features=cfg.n_features,
        mask_value=mask_value)
X_valid_data, y_valid_data = data_loader.load_data(
        datafile=filepath + 'normalized_validation.csv',
        flare_label=cfg.flare_label, series_len=1,
        start_feature=start_feature, n_features=cfg.n_features,
        mask_value=mask_value)
X_test_data, y_test_data = data_loader.load_data(
        datafile=filepath + 'normalized_testing.csv',
        flare_label=cfg.flare_label, series_len=1,
        start_feature=start_feature, n_features=cfg.n_features,
        mask_value=mask_value)
X_train_data = np.reshape(X_train_data, (len(X_train_data), cfg.n_features))
X_valid_data = np.reshape(X_valid_data, (len(X_valid_data), cfg.n_features))
X_test_data = np.reshape(X_test_data, (len(X_test_data), cfg.n_features))
# combine train and validation
X = np.concatenate((X_train_data, X_valid_data))
y = np.concatenate((y_train_data, y_valid_data))
y = data_loader.label_transform(y)

'''
Univariate Feature selection
'''
score_functions = {'F-score': f_classif,
                   'Mutual-Information': mutual_info_classif}
for scorer, score_func in score_functions.items():
    print(scorer, score_func)
    selector = SelectKBest(score_func, k=40)
    selected_features = selector.fit_transform(X_train_data, y_train_tr)
    plt.bar(data_loader.get_feature_names(
            filepath + 'normalized_training.csv')[5:], selector.scores_)
    plt.title('{} vs. Features'.format(scorer))
    plt.xlabel('Features')
    plt.xticks(rotation=90)
    plt.ylabel('Importance')
    plt.show()
    f_score_indexes = (-selector.scores_).argsort()[:40]

    sort_df = pd.concat([pd.DataFrame(data_loader.get_feature_names(
            filepath + 'normalized_training.csv')[5:], columns=['Features']),
                         pd.DataFrame(
            selector.scores_, columns=['Importance'])], axis=1).sort_values(
            by='Importance', ascending=False).reset_index()
    sns.barplot(x='Features', y='Importance', data=sort_df, order=sort_df[
        'Features'])
    plt.xticks(rotation=90)
    plt.title('Sorted {} vs. Features'.format(scorer))
    plt.show()

'''
Linear SVC optimization
'''
params = {'C': [0.001]}
clf = LinearSVC(penalty="l1", dual=False, verbose=1, max_iter=10000,
                class_weight='balanced')
grid = GridSearchCV(clf, param_grid=params,
                    cv=5,
                    return_train_score=True,
                    scoring=make_scorer(balanced_accuracy_score,
                                        **{'adjusted': True}))
grid.fit(X, y)
print("Best parameters set found on development set:")
print(grid.best_params_)
print("Grid scores on development set:")
means = grid.cv_results_['mean_test_score']
stds = grid.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, grid.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

print("Detailed classification report:")
print("The model is trained on the full development set.")
print("The scores are computed on the full evaluation set.")
y_true, y_pred = y_test_tr, grid.predict(X_test_data)
print(classification_report(y_true, y_pred))
average_tss = balanced_accuracy_score(y_test_tr, y_pred, adjusted=True)
print('Average tss score: {0:0.2f}'.format(
      average_tss))


'''
Crossval from feature selection
'''
cs = l1_min_c(X, y)  # minimum C-value
clf = Pipeline([('anova', SelectKBest(f_classif)),
                ('svc', LinearSVC(C=0.001, penalty="l1", dual=False, verbose=1,
                                  max_iter=10000,
                                  class_weight='balanced'))])

# #############################################################################
# Plot the cross-validation score as a function of number of features
score_means = list()
score_stds = list()
n_features = np.arange(40, 10, -1)

for k_feature in n_features:
    clf.set_params(anova__k=k_feature)
    this_scores = cross_val_score(clf, X, y, cv=5,
                                  scoring=make_scorer(balanced_accuracy_score,
                                                      **{'adjusted': True}))
    score_means.append(this_scores.mean())
    score_stds.append(this_scores.std())

plt.errorbar(n_features, score_means, np.array(score_stds))
plt.title(
    'Performance of the SVM-Anova varying the number of features selected')
plt.xticks(np.linspace(100, 0, 11, endpoint=True))
plt.xlabel('Number of Features')
plt.ylabel('TSS')
plt.axis('tight')
plt.savefig(drop_path + "SVM_CV.png")
plt.show()


'''
Recursive Feature Elimination
'''
clf = LinearSVC(penalty="l1", dual=False, verbose=1, max_iter=10000,
                class_weight='balanced', random_state=1, C=0.001)
clf.fit(X_train_data, y_train_tr)

y_score = clf.predict(X_valid_data)
average_tss = balanced_accuracy_score(y_valid_tr, y_score, adjusted=True)
print('Average tss score: {0:0.2f}'.format(
      average_tss))

rfe_selector = RFE(clf, 20)
rfe_selector = rfe_selector.fit(X_train_data, y_train_tr)

rfe_values = rfe_selector.get_support()
rfe_indexes = np.where(rfe_values)[0]

rfe_list = [feature_names[i+5] for i in rfe_indexes]

rfe_df = pd.concat([pd.Series(rfe_indexes), pd.Series(rfe_list)], axis=1)
print(tabulate(rfe_df.T, headers="keys",
               tablefmt="github", showindex=False))
