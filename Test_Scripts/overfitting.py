from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif, mutual_info_classif, chi2
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
import numpy as np
from data_loader import data_loader
import pandas as pd
from tabulate import tabulate

'''
F-score selection
Find top features based on f score
'''
filepath = './Data/Liu/' + 'M5' + '/'
X_train_data, y_train_data = data_loader.load_data(
        datafile=filepath + 'normalized_training.csv',
        flare_label=cfg.flare_label, series_len=1,
        start_feature=start_feature, n_features=cfg.n_features,
        mask_value=mask_value)
X_train_data = np.reshape(X_train_data, (len(X_train_data), cfg.n_features))
selector = SelectKBest(f_classif, k=20)
selected_features = selector.fit_transform(X_train_data, y_train_tr)
plt.plot(data_loader.get_feature_names(
        filepath + 'normalized_training.csv')[5:], selector.scores_)
plt.title('ANOVA F-score vs. Features')
plt.xlabel('Features')
plt.xticks(rotation=90)
plt.ylabel('Importance')
plt.show()
f_score_indexes = (-selector.scores_).argsort()[:40]


'''
Recursive Feature Elimination
'''
clf = LinearSVC(C=0.01, penalty="l1", dual=False, verbose=1)
clf.fit(X_train_data, y_train_tr)

rfe_selector = RFE(clf, 20)
rfe_selector = rfe_selector.fit(X_train_data, y_train_tr)

rfe_values = rfe_selector.get_support()
rfe_indexes = np.where(rfe_values)[0]

rfe_list = [feature_names[i+5] for i in rfe_indexes]

rfe_df = pd.concat([pd.Series(rfe_indexes), pd.Series(rfe_list)], axis=1)
print(tabulate(rfe_df.T, headers="keys",
               tablefmt="github", showindex=False))
