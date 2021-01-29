import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import pandas as pd
import torch
import re
import main_TCN_Liu
from tabulate import tabulate
import seaborn as sns
import scipy
import os
import wandb

from utils import math_stuff

sharps = ['USFLUX', 'SAVNCPP', 'TOTPOT', 'ABSNJZH', 'SHRGT45', 'AREA_ACR',
          'R_VALUE', 'TOTUSJH', 'TOTUSJZ', 'MEANJZH', 'MEANJZD', 'MEANPOT',
          'MEANSHR', 'MEANALP', 'MEANGAM', 'MEANGBZ', 'MEANGBT',
          'MEANGBH', ]  # 18
lorentz = ['TOTBSQ', 'TOTFX', 'TOTFY', 'TOTFZ', 'EPSX', 'EPSY',
           'EPSZ']  # 7
history_features = ['Bdec', 'Cdec', 'Mdec', 'Xdec', 'Edec', 'logEdec',
                    'Bhis', 'Chis', 'Mhis', 'Xhis', 'Bhis1d', 'Chis1d',
                    'Mhis1d', 'Xhis1d', 'Xmax1d']  # 15
# custom_features = ['TOTUSJH', 'TOTUSJZ','ABSNJZH', 'R_VALUE', 'Chis','Mhis']
custom_features = ['TOTUSJH', 'TOTUSJZ','ABSNJZH', 'SHRGT45', 'MEANSHR', 'MEANGAM']
feature_list = custom_features
# feature_list = feature_names[5:]
# feature_list = sharps+lorentz+history_features
# attr_name_list = ["Integrated Gradients", "DeepLIFT",
#                   "Input x Gradient", "Ablation",
#                   "Shapley Value Sampling"]
attr_name_list = ["Shapley Value Sampling"]
NOAA = 12241
flaretime = 87
directory = f"../saved/results/attribution/MLP/{NOAA}/"
drop_path = os.path.expanduser(f'~/Dropbox/_Meesters/figures/attribution/{NOAA}/')
if not os.path.exists(drop_path):
    os.makedirs(drop_path)
fig, axes = plt.subplots(10, 4, figsize=(15, 20), sharex=True)
axes = axes.reshape(-1)
df_list = []
e = pd.DataFrame()
# df_dict = {'IntegratedGradients': e, 'DeepLIFT': e,
#            'InputxGradient': e, 'Ablation': e,
#            'ShapleyValueSampling': e}
df_dict = {'ShapleyValueSampling': e}

for j, attr in enumerate(attr_name_list):
    name = attr_name_list[j].replace(' ', '')
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.startswith(name):
            print(filename)
            df = pd.read_csv(directory + filename)
            old_df = df_dict[name]
            df_dict[name] = pd.concat([old_df, df])
        else:
            continue

n0, n1 = math_stuff.get_largest_primes(len(feature_list))
fig, axes = plt.subplots(n1, n0, figsize=(10,6), sharex=True)
axes = axes.reshape(-1)
for i, feature in enumerate(feature_list):
    axes[i].set(title=feature)
    for j, attr in enumerate(attr_name_list):
        attr = attr.replace(' ', '')
        axes[i].plot(df_dict[attr].index, df_dict[attr].loc[:, feature], label=attr_name_list[j])
        axes[i].axvspan(xmin=flaretime - 24,
                        xmax=flaretime, ymin=0, ymax=1,
                        alpha=0.1, color='r')
plt.legend()
fig.text(0.5, 0.0, 'Sample', ha='center')
fig.text(0, 0.5, 'Importance', va='center', rotation='vertical')
plt.tight_layout()
plt.savefig(f'{drop_path}/attr_{NOAA}_singles.pdf')
plt.show()

#
# for j, attr in enumerate(attr_name_list):
#     attr = attr.replace(' ', '')
#     print(attr)
#
#     for i, feature in enumerate(feature_list):
#         print(feature)
#
#         ax = axes[i]
#
#         g = sns.lineplot(x=df_dict[attr].index, y=df_dict[attr].loc[:,feature],
#                          data=df_dict[attr], ci=68, label=attr_name_list[j],
#                          ax=ax)
#         ax.axvspan(xmin=127, xmax=151, ymin=0, ymax=1, alpha=0.1, color='r')
#         ax.set_title(feature)
#         ax.set_ylabel('')
#         ax.set_xlabel('')
#         ax.get_legend().set_visible(False)
#         ax.grid(True)
#         plt.tight_layout()
# plt.tight_layout()
# fig.text(0.5, 0.0, 'Sample', ha='center')
# fig.text(0, 0.5, 'Importance', va='center', rotation='vertical')
