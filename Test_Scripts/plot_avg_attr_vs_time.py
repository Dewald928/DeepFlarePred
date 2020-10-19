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
listofuncorrfeatures = ['TOTUSJH', 'SAVNCPP', 'ABSNJZH', 'TOTPOT',
                        'AREA_ACR', 'Cdec', 'Chis', 'Edec', 'Mhis',
                        'Xmax1d', 'Mdec', 'MEANPOT', 'R_VALUE', 'Mhis1d',
                        'MEANGAM', 'TOTFX', 'MEANJZH', 'MEANGBZ', 'TOTFZ',
                        'TOTFY', 'logEdec', 'EPSZ', 'MEANGBH', 'MEANJZD',
                        'Xhis1d', 'Xdec', 'Xhis', 'EPSX', 'EPSY', 'Bhis',
                        'Bdec', 'Bhis1d'] # 32
all_f = sharps+lorentz+history_features
bad_features = ['MEANPOT', 'Mhis1d', 'Edec', 'Xhis1d', 'Bdec','Bhis',
                'Bhis1d']
# feature_list = [x for x in all if x not in bad_features] #
feature_list = feature_names[5:]
# feature_list = all_f

directory = "../saved/results/attribution/MLP/"
fig, axes = plt.subplots(10, 4, figsize=(15, 20), sharex=True)
axes = axes.reshape(-1)
df_list = []
e = pd.DataFrame()
df_dict = {'Saliency': e, 'IntegratedGradients': e, 'DeepLIFT': e,
           'InputxGradient': e, 'GuidedBackprop': e, 'Ablation': e,
           'ShapleyValueSampling': e}

for j, attr in enumerate(attrs_list):
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

for j, attr in enumerate(attr_name_list):
    attr = attr.replace(' ', '')
    print(attr)

    for i, feature in enumerate(all_f):
        print(feature)

        ax = axes[i]

        g = sns.lineplot(x=df_dict[attr].index, y=df_dict[attr].loc[:,feature],
                         data=df_dict[attr], ci=68, label=attr_name_list[j],
                         ax=ax)
        ax.axvspan(xmin=127, xmax=151, ymin=0, ymax=1, alpha=0.1, color='r')
        # ax.set_xscale("log")
        ax.set_title(feature)
        ax.set_ylabel('Importance')
        ax.set_xlabel('Sample')
        ax.get_legend().set_visible(False)
        ax.grid(True)
        plt.tight_layout()
plt.legend(title='Attribution method', loc='upper left', fontsize='xx-small')
plt.savefig('../saved/results/attribution/MLP/atrr_avg_MLP.pdf')
plt.show()


