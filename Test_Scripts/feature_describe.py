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
drop_path = os.path.expanduser('~/Dropbox/_Meesters/figures/feature_boxplots/')

datasets = ['z_minmax_all', 'z_minmax_train', 'z_train', 'z_p_transformed']
describe_datasets = {'z_minmax_all': [], 'z_minmax_train': [], 'z_train': [], \
                     'z_p_transformed': []}
for dataset in datasets:
    filepath = f"../Data/Liu/{dataset}/"
    df_train = pd.read_csv(filepath + 'normalized_training.csv')
    df_val = pd.read_csv(filepath + 'normalized_validation.csv')
    df_test = pd.read_csv(filepath + 'normalized_testing.csv')
    df = df_train.loc[:, sharps + lorentz + history_features]

    describe_datasets[dataset] = df

    boxplot = df.boxplot(showfliers=False, vert=True, figsize=(10, 5),
                         showmeans=True, widths=0.6, meanline=True, rot=90,
                         meanprops={'color': 'r'})
    plt.axvline(x=26 - 0.5, ymin=0, ymax=1, color='m')

    # plt.ylim(-4,4)
    plt.title(f"Boxplot of all features. Train set. ({dataset})")
    plt.tight_layout()
    plt.savefig(drop_path + f"{dataset}.png")
    plt.show()

# quantify absolute difference between datasets
diff_Liu_Liu_train = np.abs(
    describe_datasets['z_minmax_all'] - describe_datasets['z_minmax_train'])

boxplot = diff_Liu_Liu_train.boxplot(showfliers=False, vert=True,
                                     figsize=(10, 5), showmeans=True,
                                     widths=0.6, meanline=True, rot=90,
                                     meanprops={'color': 'r'})
plt.axvline(x=26-0.5, ymin=0, ymax=1, color='m')

# plt.ylim(-4,4)
plt.title(f"Boxplot of absolute difference between (z_minmax_all) and (z_minmax_train)")
plt.tight_layout()
plt.savefig(drop_path + f"diff_train.png")
plt.show()

diff_Liu_train_Liu_z = np.abs(
    describe_datasets['z_minmax_train'] - describe_datasets['z_train'])

boxplot = diff_Liu_train_Liu_z.boxplot(showfliers=False, vert=True,
                                       figsize=(10, 5), showmeans=True,
                                       widths=0.6, meanline=True, rot=90,
                                       meanprops={'color': 'r'})
plt.axvline(x=26-0.5, ymin=0, ymax=1, color='m')

# desc_df = diff_Liu_Liu_train.describe().T
# desc_df = desc_df.loc[:, ['mean', 'std', '50%']]
# desc_df.columns = ['mean', 'std', 'median']
# print(tabulate(desc_df, tablefmt='github',
#                headers='keys', floatfmt=".3f"))
# plt.ylim(-4,4)
plt.title(f"Boxplot of absolute difference between (z_minmax_train) and ("
          f"z_train)")
plt.tight_layout()
plt.savefig(drop_path + f"diff_z.png")
plt.show()

diff_Liu_Liu_z = np.abs(
    describe_datasets['z_minmax_all'] - describe_datasets['z_train'])

boxplot = diff_Liu_Liu_z.boxplot(showfliers=False, vert=True,
                                       figsize=(10, 5), showmeans=True,
                                       widths=0.6, meanline=True, rot=90,
                                       meanprops={'color': 'r'})
# desc_df = diff_Liu_Liu_z.describe().T
# desc_df = desc_df.loc[:, ['mean', 'std', '50%']]
# desc_df.columns = ['mean', 'std', 'median']
# print(tabulate(desc_df, tablefmt='github',
#                headers='keys', floatfmt=".3f"))
# plt.ylim(-4,4)
plt.axvline(x=26-0.5, ymin=0, ymax=1, color='m')
plt.title(f"Boxplot of absolute difference between (z_minmax_all) and ("
          f"z_train)")
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
