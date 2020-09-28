import wandb
import matplotlib.pyplot as plt
import numpy as np
import skorch
from model import mlp
import torch
from data_loader import data_loader
import pandas as pd
import torch
import re
from tabulate import tabulate
import seaborn as sns
import scipy
import os

# 1. Get model
# 2. Get datasets
#         2.1. get flux
#         2.2. date
# 3. infer model for probability
# 4. plot prob over flux (with label colour)?
#         4.1 for each AR

# model already trained

dump_path = '../saved/figures/dump/'
filepath = '../Data/Liu/z_train/'
df_train = pd.read_csv(filepath + 'normalized_training.csv')
df_val = pd.read_csv(filepath + 'normalized_validation.csv')
df_test = pd.read_csv(filepath + 'normalized_testing.csv')
# Get flux and dates
df = pd.concat([df_train], axis=0)
df['flux'] = df['flare'].apply(data_loader.flare_to_flux)
df['date'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%dT%H:%M:%S.%fZ')
df = df.sort_values(by=['NOAA', 'timestamp'])

# Get noaa AR with >M5
m5_flares = df[df['label'].str.match('Positive')]
m5_flared_NOAA = m5_flares['NOAA'].unique()
m5_flares_data = df[df['NOAA'].isin(m5_flared_NOAA)]


# Predict probabilites
y_proba = net.predict_proba(inputs)
df['prob'] = y_proba[:,1]


# Plot flux and probability per AR
for i, noaa in enumerate(m5_flared_NOAA):
    print(noaa)
    df_ar = df[df['NOAA']==noaa]
    ax = df_ar.plot(x="date", y="prob", legend=False)
    plt.ylim(0,1)
    ax2 = ax.twinx()
    df_ar.plot(x="date", y="flux", ax=ax2, color="r", legend=False)
    plt.yscale('log')
    ax.figure.legend()
    plt.title(f'NOAA: {noaa}')
    plt.show()
