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
batch_list = [512, 2048, 4096, 8192, 8192*2, 65536]
# batch_list = [65536]
# dataset_list = ['Liu/z_train/', 'Liu/z_minmax_train/', 'Liu/z_minmax_all/']
dataset_list = ['Liu/z_train_relabelled/']
dropout_list = [0.5,0.6,0.7,0.8,0.9]
ksize_list = [3, 7, 13]
epochs_list = [50, 100, 200, 400]
wd_list = [0, 0.001, 0.01]
hidden_units_list = [100]

filename = 'MLP_1_100.csv'
pathname = os.path.expanduser('~/Dropbox/_Meesters/figures/MLP/Grid_Search/Relabelled/')
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
    # Get runs from a specific sweep
    sweep0 = api.runs(path="dewald123/liu_pytorch_MLP",
                      filters={'sweep': "6g21f6m2"}, per_page=1200)
    sweep1 = api.runs(path="dewald123/liu_pytorch_MLP",
                      filters={'sweep': "jjeojuaa"}, per_page=1200)
    sweep2 = api.runs(path="dewald123/liu_pytorch_MLP",
                      filters={'sweep': "tkcwt3wf"}, per_page=1200)
    sweep3 = api.runs(path="dewald123/liu_pytorch_MLP",
                      filters={'sweep': "rngahqgd"}, per_page=1200)
    # sweep4 = api.runs(path="dewald123/liu_pytorch_cnn",
    #                   filters={'sweep': "r5oj5sz1"}, per_page=1200)
    # sweep5 = api.runs(path="dewald123/liu_pytorch_cnn",
    #                   filters={'sweep': "hioonqpc"}, per_page=1200)
    # sweep6 = api.runs(path="dewald123/liu_pytorch_cnn",
    #                   filters={'sweep': "5ztffkt7"}, per_page=1200)
    # sweep7 = api.runs(path="dewald123/liu_pytorch_cnn",
    #                   filters={'sweep': "dxoynpru"}, per_page=1200)


    sweeps = [sweep0, sweep1, sweep2, sweep3]

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


get_wandb_runs(all_df)  # uncomment to get runs

# ===============================================================
# list of df from groups
all_df = pd.read_csv(pathname + filename) if os.path.isfile(
    pathname + filename) is True else pd.DataFrame()

# =============================================================
# plot graphs
fig, axes = plt.subplots(6, 5, figsize=(20,20), sharey=True)
i = 0
for batch in batch_list:
    # fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharey=True)
    # i=0
    # for batch_size in batch_list:
    # for wd in wd_list:

    for dropout in dropout_list:
        # filter_df = all_df[(all_df['dataset'] == dataset) & (
        #         all_df['batch_size'] == batch_size)]
        # filter_df = all_df[(all_df['dataset'] == dataset)]
        # filter_df = all_df[(all_df['dropout'] == dropout) & (
        #             all_df['weight_decay'] == wd)]
        filter_df = all_df[
            (all_df['dropout'] == dropout)
            & (all_df['batch_size'] == batch)]

        # filter_df = all_df[(all_df['dropout'] == dropout) & (all_df[
        #     'learning_rate']<0.8)]

        ax = axes.flat[i]
        # ax = axes

        sns.lineplot(x='learning_rate', y='Best_Train_TSS', data=filter_df,
                     ci=68, color='b', label='Train', ax=ax, marker='o')
        sns.lineplot(x='learning_rate', y='Best_Validation_TSS',
                     data=filter_df, ci=68, ax=ax, color='g',
                     label='Validation', marker='o')
        # sns.lineplot(x='learning_rate', y='Test_TSS', data=filter_df, ci=68,
        #              ax=ax, color='y', label='Test', marker='o')
        ax.set_xscale("log")
        ax.set_title(f"Dropout: {dropout}. Batch size: {batch}")
        # ax.set_title(f"Scaling strategy: {dataset[4:-1]}")
        # ax.set_title(f"Batch size: {batch_size}")
        # ax.set_title(f"Dropout: {dropout}. Weight decay: {wd}")
        ax.set_ylabel('TSS')
        ax.set_xlabel('Learning Rate')
        ax.grid(True)
        plt.tight_layout()
        i += 1
fig.tight_layout()
plt.savefig(pathname + f"TSS_vs_LR_{filename}.pdf")
plt.show()
