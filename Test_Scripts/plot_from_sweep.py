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
# batch_list = [256, 1024, 8192, 65536]
batch_list = [65536]
dataset_list = ['Liu/z_train/', 'Liu/z_minmax_train/', 'Liu/z_minmax_all/']
# dataset_list = ['Liu/z_train/']
dropout_list=[0.4,0.8,0.9,0.95]
epochs_list=[50,100, 200, 400]
# wd_list = [0, 0.001, 0.01]
hidden_units_list = [10, 25, 50, 100]

filename = 'MLP_GS_40f_62.csv'
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


def get_wandb_runs(all_df):
    # list of sweeps
    # sapcuf56
    # 919m8v0h
    # Get runs from a specific sweep
    sweep0 = api.runs(path="dewald123/liu_pytorch_MLP",
                      filters={'sweep': "29zgmi9o"}, per_page=1200)
    sweep1 = api.runs(path="dewald123/liu_pytorch_MLP",
                      filters={'sweep': "27tkmvk3"}, per_page=1200)
    sweep2 = api.runs(path="dewald123/liu_pytorch_MLP",
                      filters={'sweep': "c5qzs1sj"}, per_page=1200)
    sweep3 = api.runs(path="dewald123/liu_pytorch_MLP",
                      filters={'sweep': "3wkyzs3k"}, per_page=1200)
    # sweep1 = api.sweep("dewald123/liu_pytorch_MLP/2yk81dhq")

    sweeps = [sweep0, sweep1]

    for sweep in sweeps:
        summary_list = []
        config_list = []
        name_list = []
        extra_df = []
        sweep_df = pd.DataFrame()
        i = 0
        # history = sweep[i].history(samples=2000)
        for run in sweep:
            print(i)
            # run.summary are the output key/values like accuracy.  We
            # call ._json_dict to omit large files
            if run.state != 'finished':
                continue
            # val_tss = run.history()["Validation_TSS"].dropna().reset_index(
            #     drop=True)
            # train_TSS = run.history()["Training_TSS"].dropna().reset_index(
            #     drop=True)
            # best_train_tss = train_TSS[val_tss.idxmax()]
            # extra_df.append(best_train_tss)
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
        sweep_df = pd.concat([name_df, config_df, summary_df],
                             axis=1)  # remove/add here too
        all_df = pd.concat([all_df, sweep_df])

    # all_df = pd.concat([all_df, all_runs])
    # dataframe to csv
    all_df.to_csv(pathname + filename, index=False)


# get_wandb_runs(all_df)  # uncomment to get runs

# ===============================================================
# list of df from groups
all_df = pd.read_csv(pathname + filename) if os.path.isfile(
    pathname + filename) is True else pd.DataFrame()

# =============================================================
# plot graphs
fig, axes = plt.subplots(1,3, figsize=(10, 5), sharey=True)
i = 0
for dataset in dataset_list:
    # fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharey=True,
    #                          constrained_layout=True)

    # for hidden in hidden_units_list:
        # filter_df = all_df[(all_df['batch_size'] == batch_size) & (
        #         all_df['dataset'] == dataset)]
    filter_df = all_df[(all_df['dataset'] == dataset)]

    ax = axes.flat[i]
    # ax = axes

    sns.lineplot(x='learning_rate', y='Best_Train_TSS', data=filter_df,
                 ci=68, color='b', label='Train', ax=ax)
    sns.lineplot(x='learning_rate', y='Best_Validation_TSS',
                 data=filter_df, ci=68, ax=ax, color='g',
                 label='Validation')
    sns.lineplot(x='learning_rate', y='Test_TSS', data=filter_df, ci=68,
                 ax=ax, color='y', label='Test')
    ax.set_xscale("log")
    ax.set_title(f"Dataset: {dataset[4:-1]}")
    ax.set_ylabel('TSS')
    ax.grid(True)
    plt.tight_layout()
    i += 1
fig.tight_layout()
plt.savefig(pathname + f"TSS_vs_LR_62.png")
plt.show()