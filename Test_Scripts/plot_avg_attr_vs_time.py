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


def sort_according(old_feature_list, new_feature_list, axes):
    # get axies title, current pos
    # find in new list, idx
    # swap axes
    # update lists
    axes_orig = axes.copy()
    old_list = old_feature_list.copy()
    for i, feature in enumerate(new_feature_list):
        print(axes[i].get_title())
        new_idx = next((j for j, x in enumerate(old_list) if
                        x==feature), None)
        # print(feature, new_idx, old_list[i])
        temp = old_list[i]
        old_list[i] = new_feature_list[i]
        old_list[new_idx] = temp

        temp = axes[i]
        axes[i] = axes[new_idx]
        axes[new_idx] = temp
        print(axes[i].get_title())
        # pos1 = axes[i].get_position()
        # axes[i].set_position(axes[new_idx].get_position())
        # axes[new_idx].set_position(pos1)




directory = "./saved/results/attribution/"
fig, axes = plt.subplots(10, 4, figsize=(15, 20), sharex=True)
axes = axes.reshape(-1)
df_list = []
e = pd.DataFrame()
df_dict = {'Saliency': e, 'IntegratedGradients': e, 'DeepLIFT': e,
           'InputxGradient': e, 'GuidedBackprop': e, 'Occlusion': e,
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

    for i, feature in enumerate(feature_list):
        print(feature)

        ax = axes[i]

        g = sns.lineplot(x=df_dict[attr].index, y=df_dict[attr].iloc[:, i],
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
sort_according(feature_list, all_f, axes)
plt.legend()
plt.savefig('./saved/results/attribution/atrr_avg_test.png', dpi=500)
plt.show()


