import pandas as pd
import torch
import re

# from scipy.stats import norm_gen

import main_TCN_Liu
from tabulate import tabulate
import seaborn as sns
import scipy
import matplotlib.pyplot as plt
import numpy as np
import os
from data_loader import data_loader
from scipy.stats import normaltest
from scipy.stats import shapiro
from scipy.stats import anderson
from scipy.stats import norm
from scipy.stats import mannwhitneyu
from scipy.stats import kruskal
from scipy.stats import boxcox
import pingouin as pg
from sklearn.preprocessing import LabelBinarizer, power_transform, PowerTransformer

n_features = 40
start_feature = 5
mask_value = 0
feature_list = None
drop_path = os.path.expanduser('~/Dropbox/_Meesters/figures/features_inspect/')
# filepath = './Data/Krynauw/'
filepath = '../Data/Liu_z/'
df_train = pd.read_csv(filepath + 'normalized_training.csv')
df_val = pd.read_csv(filepath + 'normalized_validation.csv')
df_test = pd.read_csv(filepath + 'normalized_testing.csv')
# filepath = '../Data/Liu/M5/'
# df_train = pd.read_csv(filepath + 'training.csv')
# df_val = pd.read_csv(filepath + 'validation.csv')
# df_test = pd.read_csv(filepath + 'testing.csv')
df = df_train

onehot = LabelBinarizer()
onehot.fit(df['label'])
transformed = onehot.transform(df['label'])
labels = pd.DataFrame(transformed, columns=['labels'])
# df = pd.concat([labels, df], axis=1)
df = df.drop(['label','flare', 'timestamp', 'NOAA', 'HARP'], axis=1)

# QQ plot
x = df['Cdec']
fig, (ax1) = plt.subplots(1, 1, figsize=(9, 4))
ax1 = pg.qqplot(x, dist='norm', ax=ax1)
plt.show()

# histogram
fig, axes = plt.subplots(8, 5, figsize=(20, 20), sharey=False)
for i, ax in zip(range(n_features), axes.flat):
    sns.distplot(df.iloc[:, i:i + 1], fit=norm, ax=ax,
                 label=df.iloc[:, i:i + 1].columns)
    ax.legend()
plt.tight_layout()
plt.savefig(drop_path + "features_histogram.png")
plt.show()


# 0. homoscedasticity
print(pg.homoscedasticity(df))

# 1. Do univariate normality check
#  D’Agostino and Pearson’s
for i in range(n_features):
    stat = pg.normality(df.iloc[:, i], method='normaltest')
    print(f"{stat.index[0]} - {stat['W'][0]:.1f} - {stat['pval'][0]} -"
          f" {stat['normal'][0]}")


# 3. Do transform data with box-cox
pt = PowerTransformer(method='yeo-johnson')
transformed = pt.fit_transform(df)
df_trans = pd.DataFrame(transformed, columns=df.columns)
# test
print(pg.homoscedasticity(df_trans))

# plot transformed data
fig, axes = plt.subplots(8, 5, figsize=(20, 20), sharey=False)
for i, ax in zip(range(n_features), axes.flat):
    sns.distplot(df_trans.iloc[:, i:i + 1], fit=norm, ax=ax,
                 label=df.iloc[:, i:i + 1].columns)
    ax.legend()
plt.tight_layout()
plt.legend()
plt.savefig(drop_path + "features_hist_trans.png")
plt.show()

# QQ plot
x = df_trans['Cdec']
fig, (ax1) = plt.subplots(1, 1, figsize=(9, 4))
ax1 = pg.qqplot(x, dist='norm', ax=ax1)
plt.show()

#  D’Agostino and Pearson’s
for i in range(n_features):
    stat = pg.normality(df_trans.iloc[:, i], method='normaltest')
    print(f"{stat.index[0]} - {stat['W'][0]:.1f} - {stat['pval'][0]:.2f} -"
          f" {stat['normal'][0]}")


# 4. Kruskal-Wallis non-parametric ANOVA
# non-parametric
# data = df
# args = [data[col] for col in data.columns]
# stat, p = kruskal(*args)
# print('Statistics=%.3f, p=%.3f' % (stat, p))
# # interpret
# alpha = 0.05
# if p > alpha:
#     print('Same distributions (fail to reject H0)')
# else:
#     print('Different distributions (reject H0)')
#


# final qqplot comparison
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
x0 = df['Cdec']
x1 = df_trans['Cdec']
pg.qqplot(x0, dist='norm', ax=axes[0])
pg.qqplot(x1, dist='norm', ax=axes[1])
axes[0].title.set_text('Q-Q Plot, Cdec before transform')
axes[1].title.set_text('Q-Q Plot, Cdec after transform')
plt.savefig(drop_path+"QQ_Cdec.png")
plt.show()

# save new dataset
trans_train = pt.transform(df_train.iloc[:,5:])
trans_val = pt.transform(df_val.iloc[:,5:])
trans_test = pt.transform(df_test.iloc[:,5:])

df_trans_train = pd.DataFrame(trans_train, columns=df.columns)
df_trans_val = pd.DataFrame(trans_val, columns=df.columns)
df_trans_test = pd.DataFrame(trans_val, columns=df.columns)

t_train = pd.concat([df_train.iloc[:,:5], df_trans_train], axis=1)
t_val = pd.concat([df_val.iloc[:,:5], df_trans_val], axis=1)
t_test = pd.concat([df_test.iloc[:,:5], df_trans_test], axis=1)

new_path = '../Data/Liu_transformed/'
t_train.to_csv(new_path + 'transformed_training.csv', index=False)
t_val.to_csv(new_path + 'transformed_validation.csv', index=False)
t_test.to_csv(new_path + 'transformed_testing.csv', index=False)
