import numpy as np
import pandas as pd
import wandb
import os.path
import seaborn as sns
import matplotlib.pyplot as plt
api = wandb.Api()

# # Get runs from a specific sweep
# sweep0 = api.sweep("dewald123/liu_pytorch_MLP/xris1c8z")
# sweep1 = api.sweep("dewald123/liu_pytorch_MLP/0iihk32s")
# sweep2 = api.sweep("dewald123/liu_pytorch_MLP/1c6ig5mc")
# sweep = api.sweep("dewald123/liu_pytorch_MLP/1c6ig5mc")
# sweeps = [sweep0, sweep1, sweep2]

all_runs = pd.read_csv('all_runs.csv') if os.path.isfile(
    'all_runs.csv') is True else pd.DataFrame()

# for sweep in sweeps:
#     # if the file is already written to, read it
#     all_runs = pd.read_csv('all_runs.csv') if os.path.isfile(
#         'all_runs.csv') is True else pd.DataFrame()
#
#     for i in range(len(sweep.runs)):
#         w = 3
#         run_data = pd.DataFrame()
#         # wandb.init(project='liu_pytorch_MLP', entity='dewald123',
#         #            name=sweep.runs[i].name, id=sweep.runs[i].id, resume=True,
#         #            config=sweep.runs[i].config)
#         # sweep.runs[i].update()
#         # get val tss for run
#         history = sweep.runs[i].history(samples=1000)
#         val_tss = history["Validation_TSS"].dropna().reset_index(drop=True)
#         run_id = sweep.runs[i].id
#         run_config = sweep.runs[i].config
#         run_config = {k: run_config[k] for k in
#                       run_config.keys() & {'layers', 'hidden_units', 'batch_size',
#                                            'learning_rate', 'seed',
#                                            'n_features'}}
#         run_config.update({'id': run_id})
#         run_config = pd.DataFrame(run_config, index=range(1))
#         run_config = run_config.append([run_config] * (len(val_tss)-1),
#                                      ignore_index=True)
#
#         # calculate moving avg and std
#         moving_avg = val_tss.expanding(min_periods=1).mean()
#         moving_std = val_tss.expanding(min_periods=1).std()
#         # window averages
#         moving_avg_w = val_tss.rolling(window=w).mean()
#         moving_std_w = val_tss.rolling(window=w).std()
#         moving_avg_g = pd.Series(np.gradient(moving_avg_w))
#         moving_std_g = pd.Series(np.gradient(moving_std_w))
#         moving_std_w = moving_std_w.fillna(0)
#         moving_avg_w = moving_avg_w.fillna(0)
#         moving_std_g = moving_std_g.fillna(0)
#         moving_avg_g = moving_avg_g.fillna(0)
#         moving_std = moving_std.fillna(0)
#         moving_avg = moving_avg.fillna(0)
#         moving_avg.name = 'moving_avg'
#         moving_avg_w.name = 'moving_avg_w'
#         moving_avg_g.name = 'moving_avg_g'
#         moving_std.name = 'moving_std'
#         moving_std_w.name = 'moving_std_w'
#         moving_std_g.name = 'moving_std_g'
#         val_tss = val_tss.to_frame()
#
#         val_tss['epoch'] = np.arange(len(val_tss))
#
#         # create dataframe from the run
#         run_data = pd.concat([run_config, val_tss, moving_avg, moving_std,
#                               moving_avg_w, moving_std_w, moving_avg_g, moving_std_g],
#                              axis=1)
#
#         # append run to other runs
#         all_runs = pd.concat([all_runs, run_data], axis=0, sort=False)
#         print(i)
#
#     # dataframe to csv
#     all_runs.to_csv('all_runs.csv')
#




# seaborn plots
layers = 1
hidden_units = 40
batch_size = 4096
learning_rate = 1e-3
seed = 15

fig, ax1 = plt.subplots(figsize=(10, 6))
color = 'tab:blue'
ax1.set_title('Validation TSS and standard deviations', fontsize=16)
ax1.set_xlabel('Epoch', fontsize=16)
ax1.set_ylabel('Validation TSS', fontsize=16, color=color)
ax1.set_ylim(0, 1)
ax2 = sns.lineplot(x='epoch', y='Validation_TSS', color=color,
                   data=all_runs[(all_runs['layers'] == layers)
                                 & (all_runs['hidden_units'] == hidden_units)
                                 & (all_runs['batch_size'] == batch_size)
                                 & (all_runs['learning_rate'] == learning_rate)
                                 & (all_runs['seed'] == seed)])
# & (all_runs['seed'] == seed)
ax1.tick_params(axis='y')
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Moving STD', fontsize=16, color=color)
ax2.set_ylim(-0.01, 0.01)
ax2 = sns.lineplot(x='epoch', y=pd.Series(np.gradient(all_runs[
                                                             'moving_std'])),
                   color=color,
                   data=all_runs[(all_runs['layers'] == layers)
                                 & (all_runs['hidden_units'] == hidden_units)
                                 & (all_runs['batch_size'] == batch_size)
                                 & (all_runs['learning_rate'] == learning_rate)
                                 & (all_runs['seed'] == seed)])
ax2.tick_params(axis='y', color=color)
plt.legend(['layers:{} \nnodes:{} \nbatch_size:{} \nlearning_rate:{} \n'
           'seed:{}'
           ''.format(layers, hidden_units, batch_size, learning_rate,
                     'all')], loc=8)
plt.show()









