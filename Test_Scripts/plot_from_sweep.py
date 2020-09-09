import pandas as pd
import torch
import re
import main_TCN_Liu
from tabulate import tabulate
import seaborn as sns
import scipy
import matplotlib.pyplot as plt
import numpy as np
import os
import wandb

# Plots validation tss with se over learning rate for groups
# ==============================================================
batch_list = [256, 1024, 8192, 65536]
dataset_list = ['Liu/z_train/', 'Liu/z_minmax_train/', 'Liu/z_minmax_all/']

filename = 'MLP_GS_40f_2.csv'
pathname = os.path.expanduser('~/Dropbox/_Meesters/figures/MLP/Grid_Search/')
all_df = pd.read_csv(pathname + filename) if os.path.isfile(
    pathname + filename) is True else pd.DataFrame()
# model_type = 'MLP'  # 'TCN 'or 'MLP'
# if model_type == 'MLP':
#     HP_list = ['layers', 'hidden_units', 'batch_size', 'learning_rate',
#     'seed']
#     HP_groupby = ['layers', 'hidden_units', 'batch_size', 'learning_rate']

# ==================================================================
# Get wandb runs to df
api = wandb.Api()


def get_wandb_runs():
    # list of sweeps
    # sapcuf56
    # 919m8v0h
    # Get runs from a specific sweep
    sweep0 = api.runs(path="dewald123/liu_pytorch_MLP",
                      filters={'sweep': "919m8v0h"}, per_page=1200)
    # sweep1 = api.sweep("dewald123/liu_pytorch_MLP/2yk81dhq")

    sweeps = [sweep0]

    for sweep in sweeps:
        summary_list = []
        config_list = []
        name_list = []
        extra_df = []
        i = 0
        # history = sweep[i].history(samples=2000)
        for run in sweep:
            print(i)
            # run.summary are the output key/values like accuracy.  We
            # call ._json_dict to omit large files
            val_tss = run.history()["Validation_TSS"].dropna().reset_index(
                drop=True)
            train_TSS = run.history()["Training_TSS"].dropna().reset_index(
                drop=True)
            best_train_tss = train_TSS[val_tss.idxmax()]
            extra_df.append(best_train_tss)
            summary_list.append(run.summary._json_dict)

            # run.config is the input metrics.  We remove special values
            # that start with _.
            config_list.append(
                {k: v for k, v in run.config.items() if not k.startswith('_')})

            # run.name is the name of the run.
            name_list.append(run.name)
            i += 1

        summary_df = pd.DataFrame.from_records(summary_list)
        config_df = pd.DataFrame.from_records(config_list)
        name_df = pd.DataFrame({'name': name_list})
        # extra_df = pd.DataFrame({'Best_Train_TSS': extra_df}) # todo comment
        # out if train tss already logged
        all_df = pd.concat([name_df, config_df, summary_df],
                           axis=1)  # remove/add here too

    # dataframe to csv
    all_df.to_csv(pathname + filename, index=False)


get_wandb_runs() # uncomment to get runs

# ===============================================================
# list of df from groups
all_df = pd.read_csv(pathname + filename) if os.path.isfile(
    pathname + filename) is True else pd.DataFrame()

# =============================================================
# plot graphs

for dataset in dataset_list:
    # fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharey=True,
    #                          constrained_layout=True)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    i = 0
    for batch_size in batch_list:
        filter_df = all_df[(all_df['batch_size'] == batch_size) & (
                all_df['dataset'] == dataset)]

        ax = axes.flat[i]

        sns.lineplot(x='learning_rate', y='Best_Train_TSS', data=filter_df,
                     ci=68, color='b', label='Train', ax=ax)
        sns.lineplot(x='learning_rate', y='Best_Validation_TSS',
                     data=filter_df, ci=68, ax=ax, color='g',
                     label='Validation')
        sns.lineplot(x='learning_rate', y='Test_TSS', data=filter_df, ci=68,
                     ax=ax, color='y', label='Test')
        ax.set_xscale("log")
        ax.set_title(f"Scaling strategy: {dataset[4:-1]}. Batch size:"
                     f" {batch_size}")
        ax.set_ylabel('TSS')
        plt.tight_layout()
        i += 1
    fig.tight_layout()
    plt.savefig(pathname + f"TSS_vs_LR_{dataset[4:-1]}")
    plt.show()
