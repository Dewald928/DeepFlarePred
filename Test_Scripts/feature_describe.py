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
from sklearn.preprocessing import LabelBinarizer, power_transform, \
    PowerTransformer

n_features = 40
start_feature = 5
mask_value = 0
feature_list = None
drop_path = os.path.expanduser('~/Dropbox/_Meesters/figures/feature_boxplots/')

datasets = ['Liu_z', 'Liu/M5', 'Liu_train', 'Liu_transformed']
describe_datasets = {'Liu_z': [], 'Liu/M5': [], 'Liu_train': [], \
                     'Liu_transformed': []}
for dataset in datasets:
    filepath = f"../Data/{dataset}/"
    df_train = pd.read_csv(filepath + 'normalized_training.csv')
    df_val = pd.read_csv(filepath + 'normalized_validation.csv')
    df_test = pd.read_csv(filepath + 'normalized_testing.csv')
    df = df_train.iloc[:, 5:]
    dataset = 'Liu' if dataset == 'Liu/M5' else dataset
    describe_datasets[dataset] = df

    boxplot = df.boxplot(showfliers=False, vert=True, figsize=(10, 5),
                         showmeans=True, widths=0.6, meanline=True, rot=90,
                         meanprops={'color': 'r'})
    # plt.ylim(-4,4)
    plt.title(f"Boxplot of all features. Train set. ({dataset})")
    plt.tight_layout()
    plt.savefig(drop_path + f"{dataset}.png")
    plt.show()

# quantify absolute difference between datasets
diff_Liu_Liu_train = np.abs(
    describe_datasets['Liu'] - describe_datasets['Liu_train'])

boxplot = diff_Liu_Liu_train.boxplot(showfliers=False, vert=True,
                                     figsize=(10, 5), showmeans=True,
                                     widths=0.6, meanline=True, rot=90,
                                     meanprops={'color': 'r'})
# plt.ylim(-4,4)
plt.title(f"Boxplot of absolute difference between (Liu) and (Liu_train)")
plt.tight_layout()
plt.savefig(drop_path + f"diff_train.png")
plt.show()

diff_Liu_train_Liu_z = np.abs(
    describe_datasets['Liu_train'] - describe_datasets['Liu_z'])

boxplot = diff_Liu_train_Liu_z.boxplot(showfliers=False, vert=True,
                                       figsize=(10, 5), showmeans=True,
                                       widths=0.6, meanline=True, rot=90,
                                       meanprops={'color': 'r'})
# plt.ylim(-4,4)
plt.title(f"Boxplot of absolute difference between (Liu_train) and ("
          f"Liu_z)")
plt.tight_layout()
plt.savefig(drop_path + f"diff_z.png")
plt.show()

diff_Liu_Liu_z = np.abs(
    describe_datasets['Liu'] - describe_datasets['Liu_z'])

boxplot = diff_Liu_Liu_z.boxplot(showfliers=False, vert=True,
                                       figsize=(10, 5), showmeans=True,
                                       widths=0.6, meanline=True, rot=90,
                                       meanprops={'color': 'r'})
# plt.ylim(-4,4)
plt.title(f"Boxplot of absolute difference between (Liu) and ("
          f"Liu_z)")
plt.tight_layout()
plt.savefig(drop_path + f"diff_liu_z.png")
plt.show()


# abs sum of means
a = diff_Liu_Liu_train.describe()
print(f"Train diff: {a.loc['mean', :].abs().mean():.4f}")

b = diff_Liu_train_Liu_z.describe()
print(f"Scaler diff: {b.loc['mean', :].abs().mean():.4f}")

c = diff_Liu_Liu_z.describe()
print(f"All diff: {c.loc['mean', :].abs().mean():.4f}")

# although features have stronger effect
