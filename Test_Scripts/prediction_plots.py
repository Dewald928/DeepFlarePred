'''
Get flux and probability plots over time for each AR
'''

import wandb
import matplotlib.pyplot as plt
import numpy as np
import skorch
from model import mlp
from model import metric
import torch
from data_loader import data_loader
import pandas as pd
import torch
import re
from tabulate import tabulate
import seaborn as sns
import scipy
import os
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
# 1. Get model
# 2. Get datasets
#         2.1. get flux
#         2.2. date
# 3. infer model for probability
# 4. plot prob over flux (with label colour)?
#         4.1 for each AR

# model already trained

dump_path = os.path.expanduser(
    '~/Dropbox/_Meesters/figures/MLP/NOAA_prediction/')
if not os.path.exists(dump_path):
    os.makedirs(dump_path)
filepath = './Data/Liu/z_train/'
df_train = pd.read_csv(filepath + 'normalized_training.csv')
df_val = pd.read_csv(filepath + 'normalized_validation.csv')
df_test = pd.read_csv(filepath + 'normalized_testing.csv')
# Get flux and dates
df = pd.concat([df_train,df_val,df_test], axis=0)
df['Flux'] = df['flare'].apply(data_loader.flare_to_flux)
df['Date'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%dT%H:%M:%S.%fZ')
df = df.sort_values(by=['NOAA', 'timestamp'])


# Get noaa AR with >M5
m5_flares = df[df['label'].str.match('Positive')]
m5_flared_NOAA = m5_flares['NOAA'].unique()
m5_flares_data = df[df['NOAA'].isin(m5_flared_NOAA)]

# m_flares = df[df['flare'].str.contains('M')]
# m_flares_NOAA = m_flares['NOAA'].unique()
# m_flares_data = df[df['NOAA'].isin(m_flares_NOAA)]
#
# x_flares = df[df['flare'].str.contains('X')]
# x_flares_NOAA = x_flares['NOAA'].unique()
# x_flares_data = df[df['NOAA'].isin(x_flares_NOAA)]


# Predict probabilites
y_proba = metric.get_proba(model(torch.cat((X_train_data_tensor,
                                            X_valid_data_tensor,
                                            X_test_data_tensor),
                                           0).to(device)))
y_pred = metric.to_labels(y_proba)
df['Prob'] = y_proba[:,1]
df['Pred'] = y_pred[:,1]
df['Target'] = le.fit_transform(df['label'])

# Plot flux and probability per AR
for i, noaa in enumerate(m5_flared_NOAA):
    print(noaa)
    fig, axes = plt.subplots(1,2, figsize=(10,5))
    df_ar = df[df['NOAA']==noaa]
    lns1 = df_ar.plot(x="Date", y="Prob", ax=axes[0], legend=False)
    axes[0].axvspan(xmin=df_ar['Date'].iloc[1], xmax=df_ar['Date'].iloc[-1],
               ymin=0.5,
               ymax=1, alpha=0.2,
               color='b')
    axes[0].set(ylabel='Probability')
    axes[0].set(xlabel='Date')
    axes[0].set(ylim=(0,1))
    ax2 = axes[0].twinx()
    lns2 = df_ar.plot(x="Date", y="Flux", ax=ax2, color="r", legend=False)
    ax2.axvspan(xmin=df_ar['Date'].iloc[1], xmax=df_ar['Date'].iloc[-1],
               ymin=0.68,
               ymax=1, alpha=0.2,
               color='r')
    ax2.set(yscale='log')
    ax2.set(ylim=(1e-7, 1e-3))
    ax2.set(ylabel='Flux')
    h1, l1 = axes[0].get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    axes[0].legend(h1 + h2, l1 + l2, loc=4, bbox_to_anchor=(1.15, -0.25))
    ax2.set(title=f'NOAA: {noaa}')
    # axes 2
    df_ar.plot(x="Date", y="Pred", ax=axes[1], legend=False)
    df_ar.plot(x="Date", y="Target", ax=axes[1], color='r', legend=False)
    axes[1].legend(loc=4,bbox_to_anchor=(1.15, -0.25))
    axes[1].set(title=f'NOAA: {noaa}')
    axes[1].set(ylabel='Prediction')
    plt.tight_layout()
    plt.savefig(dump_path + f"NOAA_{noaa}.png")
    plt.show()

#
# # Plot flux and probability per AR
# for i, noaa in enumerate(m5_flared_NOAA):
#     print(noaa)
#     df_ar = df[df['NOAA']==noaa]
#     ax = df_ar.plot(x="date", y="prob", legend=False)
#     ax.axvspan(xmin=df_ar['date'].iloc[1], xmax=df_ar['date'].iloc[-1],
#                ymin=0.5,
#                ymax=1, alpha=0.2,
#                color='b')
#     plt.ylabel('Probability')
#     plt.xlabel('Date')
#     plt.ylim(-0.05,1.05)
#     ax2 = ax.twinx()
#     df_ar.plot(x="date", y="flux", ax=ax2, color="r", legend=False)
#     ax2.axvspan(xmin=df_ar['date'].iloc[1], xmax=df_ar['date'].iloc[-1],
#                ymin=0.68,
#                ymax=1, alpha=0.2,
#                color='r')
#     plt.yscale('log')
#     plt.ylim(1e-7, 1e-3)
#     plt.ylabel('Flux')
#     ax.figure.legend(loc=4)
#     plt.title(f'NOAA: {noaa}')
#     plt.savefig(dump_path + f"Prob_NOAA_{noaa}.png")
#     plt.tight_layout()
#     plt.show()
#
# # Plot predicted and target per AR
# for i, noaa in enumerate(m5_flared_NOAA):
#     print(noaa)
#     df_ar = df[df['NOAA']==noaa]
#     ax = df_ar.plot(x="date", y="pred", legend=False)
#     plt.ylabel('Prediction')
#     plt.xlabel('Date')
#     plt.ylim(-0.05,1.05)
#     ax2 = ax.twinx()
#     df_ar.plot(x="date", y="target", ax=ax2, legend=False)
#     plt.ylim(-0.05, 1.05)
#     plt.ylabel('Label')
#     ax.figure.legend(loc=4)
#     plt.title(f'NOAA: {noaa}')
#     plt.savefig(dump_path + f"Pred_NOAA_{noaa}.png")
#     plt.show()