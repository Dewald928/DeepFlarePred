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
from scipy.stats import stats
from scipy.stats import skew, skewtest
from scipy.stats import norm, kurtosis, kurtosistest
import pingouin as pg
from sklearn.preprocessing import LabelBinarizer, power_transform, \
    PowerTransformer

# dagistino test

x = np.random.normal(-1, 1, size=20)
sns.distplot(x, fit=norm)
plt.show()
k2, p = stats.normaltest(x)
print(k2, p)
alpha = 1e-3
if p < alpha:  # null hypothesis: x comes from a normal distribution
    print("The null hypothesis can be rejected")
else:
    print("The null hypothesis cannot be rejected")

sharps = ['USFLUX', 'SAVNCPP', 'TOTPOT', 'ABSNJZH', 'SHRGT45', 'AREA_ACR',
          'R_VALUE', 'TOTUSJH', 'TOTUSJZ', 'MEANJZH', 'MEANJZD', 'MEANPOT',
          'MEANSHR', 'MEANALP', 'MEANGAM', 'MEANGBZ', 'MEANGBT',
          'MEANGBH', ]  # 18
lorentz = ['TOTBSQ', 'TOTFX', 'TOTFY', 'TOTFZ', 'EPSX', 'EPSY', 'EPSZ']  # 7
history_features = ['Bdec', 'Cdec', 'Mdec', 'Xdec', 'Edec', 'logEdec', 'Bhis',
                    'Chis', 'Mhis', 'Xhis', 'Bhis1d', 'Chis1d', 'Mhis1d',
                    'Xhis1d', 'Xmax1d']  # 15

n_features = 40
start_feature = 5
mask_value = 0
feature_list = None
drop_path = os.path.expanduser('~/Dropbox/_Meesters/figures/features_inspect/')
# drop_path = os.path.expanduser(
#     '~/projects/DeepFlarePred/saved/feature_boxplots/')
# filepath = './Data/Krynauw/'
filepath = '../Data/Liu/z_train/'
df_train = pd.read_csv(filepath + 'normalized_training.csv')
df_val = pd.read_csv(filepath + 'normalized_validation.csv')
df_test = pd.read_csv(filepath + 'normalized_testing.csv')
# filepath = '../Data/Liu/M5/'
# df_train = pd.read_csv(filepath + 'training.csv')
# df_val = pd.read_csv(filepath + 'validation.csv')
# df_test = pd.read_csv(filepath + 'testing.csv')
df = df_train

# m5_flares = df[df['label'].str.match('Positive')]
# m5_flared_NOAA = m5_flares['NOAA'].unique()
# m5_flares_data = df[df['NOAA'].isin(m5_flared_NOAA)]
# df_train = m5_flares_data

print(tabulate(pd.DataFrame(df['Cdec'].describe()), headers='keys',
               floatfmt='.4f', tablefmt='github'))

onehot = LabelBinarizer()
onehot.fit(df['label'])
transformed = onehot.transform(df['label'])
labels = pd.DataFrame(transformed, columns=['labels'])
df_train = df_train.loc[:, sharps + lorentz + history_features]
df_val = df_val.loc[:, sharps + lorentz + history_features]
df_test = df_test.loc[:, sharps + lorentz + history_features]

# QQ plot
x = df_train['MEANGBH']
fig, (ax1) = plt.subplots(1, 1, figsize=(9, 4))
ax1 = pg.qqplot(x, dist='norm', ax=ax1)
plt.show()

# histogram
strong_normal = ['MEANGBZ', 'MEANGBT', 'MEANGBH', 'EPSZ']
weak_leptokurtic = ['MEANJZH', 'MEANJZD', 'MEANALP', 'TOTFX', 'TOTFY']
bimodal = ['EPSX', 'EPSY']
fig, axes = plt.subplots(8, 5, figsize=(20, 20), sharey=False)
for i, ax in zip(range(n_features), axes.flat):
    if df_train.iloc[:, i:i + 1].columns in strong_normal:
        colour = '#42ff4b'
    elif df_train.iloc[:, i:i + 1].columns in weak_leptokurtic:
        colour = '#ffff42'
    elif df_train.iloc[:, i:i + 1].columns in bimodal:
        colour = '#ff4242'
    else:
        colour = '#428eff'
    sns.distplot(df_train.iloc[:, i:i + 1], fit=norm, ax=ax,
                 label=df_train.iloc[:, i:i + 1].columns, color=colour)
    ax.legend()
plt.tight_layout()
plt.savefig(drop_path + "features_histogram.png")
plt.show()

# 0. homoscedasticity
print(pg.homoscedasticity(df_train))

# 1. Do univariate normality check
#  D’Agostino and Pearson’s
stat = pg.normality(df_train, method='normaltest', alpha=2e-37)
# skewness and kurtosis
skew_df = pd.DataFrame()
kurt_df = pd.DataFrame()
for i in range(n_features):
    skewness = pd.DataFrame(skew(df_train.iloc[:, i:i + 1]),
                            columns=df_train.iloc[:, i:i + 1].columns)
    skew_df = pd.concat([skew_df, skewness], axis=1)
    kurt = pd.DataFrame(kurtosis(df_train.iloc[:, i:i + 1]),
                        columns=df_train.iloc[:, i:i + 1].columns)
    kurt_df = pd.concat([kurt_df, kurt], axis=1)

stat.insert(0, 'skewness', skew_df.T)
stat.insert(1, 'kurtosis', kurt_df.T)
print(tabulate(stat, tablefmt='github', floatfmt=('.0f','.1f','.1f', '.2f',
                                                  '.0e'),
               headers='keys'))

# 3. Do transform data with box-cox
pt = PowerTransformer(method='yeo-johnson')
transformed = pt.fit_transform(df_train)
df_trans = pd.DataFrame(transformed, columns=df_train.columns)
# test
print(pg.homoscedasticity(df_trans))

# plot transformed data
strong_normal = ['MEANGBZ', 'MEANGBT', 'MEANGBH', 'EPSZ', 'R_VALUE', 'TOTFZ']
weak_leptokurtic = ['MEANJZH', 'MEANJZD', 'MEANALP', 'TOTFX', 'TOTFY']
bimodal = ['EPSX', 'EPSY']
fig, axes = plt.subplots(8, 5, figsize=(20, 20), sharey=False)
for i, ax in zip(range(n_features), axes.flat):
    if df_train.iloc[:, i:i + 1].columns in strong_normal:
        colour = '#42ff4b'
    elif df_train.iloc[:, i:i + 1].columns in weak_leptokurtic:
        colour = '#ffff42'
    elif df_train.iloc[:, i:i + 1].columns in bimodal:
        colour = '#ff4242'
    else:
        colour = '#428eff'
    sns.distplot(df_trans.iloc[:, i:i + 1], fit=norm, ax=ax,
                 label=df_trans.iloc[:, i:i + 1].columns, color=colour)
    ax.legend()
plt.tight_layout()
plt.legend()
plt.savefig(drop_path + "features_hist_trans.png")
plt.show()

# QQ plot
x = df_trans['MEANGBH']
fig, (ax1) = plt.subplots(1, 1, figsize=(9, 4))
ax1 = pg.qqplot(x, dist='norm', ax=ax1)
plt.show()

#  D’Agostino and Pearson’s
stat = pg.normality(df_trans, method='normaltest', alpha=2e-37)
# skewness and kurtosis (not be be confused with z-score of skewness)
skew_df = pd.DataFrame()
kurt_df = pd.DataFrame()
for i in range(n_features):
    skewness = pd.DataFrame(skew(df_trans.iloc[:, i:i + 1]),
                            columns=df_trans.iloc[:, i:i + 1].columns)
    skew_df = pd.concat([skew_df, skewness], axis=1)
    kurt = pd.DataFrame(kurtosis(df_trans.iloc[:, i:i + 1]),
                        columns=df_trans.iloc[:, i:i + 1].columns)
    kurt_df = pd.concat([kurt_df, kurt], axis=1)

stat.insert(0, 'skewness', skew_df.T)
stat.insert(1, 'kurtosis', kurt_df.T)
print(tabulate(stat, tablefmt='github',
               floatfmt=('.0f', '.1f', '.1f', '.2f', '.0e'), headers='keys'))

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
x0 = df_train['Cdec']
x1 = df_trans['Cdec']
pg.qqplot(x0, dist='norm', ax=axes[0])
pg.qqplot(x1, dist='norm', ax=axes[1])
axes[0].title.set_text('Q-Q Plot, Cdec before transform')
axes[1].title.set_text('Q-Q Plot, Cdec after transform')
plt.tight_layout()
plt.savefig(drop_path + "QQ_Cdec.png")
plt.show()

# save new dataset
trans_train = pt.transform(df_train)
trans_val = pt.transform(df_val)
trans_test = pt.transform(df_test)

df_trans_train = pd.DataFrame(trans_train, columns=df_train.columns)
df_trans_val = pd.DataFrame(trans_val, columns=df_train.columns)
df_trans_test = pd.DataFrame(trans_val, columns=df_train.columns)

t_train = pd.concat([df.iloc[:, :5], df_trans_train], axis=1)
t_val = pd.concat([df.iloc[:, :5], df_trans_val], axis=1)
t_test = pd.concat([df.iloc[:, :5], df_trans_test], axis=1)

new_path = '../Data/Liu/z_p_transformed/'
if not os.path.exists(new_path):
    os.makedirs(new_path)
t_train.to_csv(new_path + 'normalized_training.csv', index=False)
t_val.to_csv(new_path + 'normalized_validation.csv', index=False)
t_test.to_csv(new_path + 'normalized_testing.csv', index=False)
