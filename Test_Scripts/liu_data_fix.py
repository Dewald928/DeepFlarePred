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

drop_path = os.path.expanduser(
    '~/Dropbox/_Meesters/figures/features_inspect/')
filepath = '../Data/Liu/' + 'M5' + '/'

splits = ['training', 'validation', 'testing']

for split in splits:
    df = pd.read_csv(filepath + '{}_old.csv'.format(split))

    df = df.rename(columns={'SAVNCPP': 'USFLUX', 'ABSNJZH': 'SAVNCPP',
                            'TOTPOT': 'TOTUSJZ', 'TOTUSJZ': 'ABSNJZH',
                            'TOTBSQ': 'TOTPOT', 'USFLUX': 'AREA_ACR',
                            'AREA_ACR': 'MEANPOT', 'MEANPOT': 'R_VALUE',
                            'SHRGT45': 'MEANGAM', 'MEANSHR': 'MEANJZH',
                            'MEANJZD': 'MEANALP', 'MEANJZH': 'MEANSHR',
                            'MEANGAM': 'MEANGBZ', 'MEANALP': 'TOTBSQ',
                            'R_VALUE': 'SHRGT45', 'MEANGBZ': 'MEANJZD', })

    df.to_csv(filepath + '{}.csv'.format(split), index=False)

