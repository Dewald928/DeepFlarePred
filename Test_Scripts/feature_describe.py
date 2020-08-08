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
drop_path = os.path.expanduser('~/Dropbox/_Meesters/figures/feature_boxplots/')

datasets = ['Liu_z', 'Liu/M5', 'Liu_train']
for dataset in datasets:
    filepath = f"../Data/{dataset}/"
    df_train = pd.read_csv(filepath + 'normalized_training.csv')
    df_val = pd.read_csv(filepath + 'normalized_validation.csv')
    df_test = pd.read_csv(filepath + 'normalized_testing.csv')
    df = df_train.iloc[:,5:]

    boxplot = df.boxplot(showfliers=False, vert=False, figsize=(10,12),
                         showmeans=True, widths=0.6, meanline=True)
    plt.xlim(-4,4)
    plt.tight_layout()
    dataset = 'Liu' if dataset == 'Liu/M5' else dataset
    plt.title(f"Boxplot of all features. Train set. ({dataset})")
    plt.savefig(drop_path + f"{dataset}.png")
    plt.show()
