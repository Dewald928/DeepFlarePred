import numpy as np
import pandas as pd
import wandb
import os.path
import os
import seaborn as sns
import matplotlib.pyplot as plt
from tabulate import tabulate

api = wandb.Api()
filename = 'all_runs.csv'
pathname = os.path.expanduser(
    '~/Dropbox/_Meesters/figures/moving_std_val_tss/')
all_runs = pd.read_csv(filename) if os.path.isfile(
    filename) is True else pd.DataFrame()

def get_wandb_runs():
    # Get runs from a specific sweep
    sweep0 = api.sweep("dewald123/liu_pytorch_MLP/xris1c8z")
    sweep1 = api.sweep("dewald123/liu_pytorch_MLP/0iihk32s")
    sweep2 = api.sweep("dewald123/liu_pytorch_MLP/1c6ig5mc")
    # sweep = api.sweep("dewald123/liu_pytorch_MLP/1c6ig5mc")
    sweeps = [sweep0, sweep1, sweep2]


    for sweep in sweeps:
        # if the file is already written to, read it
        all_runs = pd.read_csv(filename) if os.path.isfile(
            filename) is True else pd.DataFrame()

        for i in range(len(sweep.runs)):
            w = 20
            run_data = pd.DataFrame()
            # wandb.init(project='liu_pytorch_MLP', entity='dewald123',
            #            name=sweep.runs[i].name, id=sweep.runs[i].id,
            resume=True,
            #            config=sweep.runs[i].config)
            # sweep.runs[i].update()
            # get val tss for run
            history = sweep.runs[i].history(samples=1000)
            val_tss = history["Validation_TSS"].dropna().reset_index(drop=True)
            test_tss = history["Test_TSS"].dropna().reset_index(drop=True)
            test_tss = test_tss.append([test_tss]*(len(val_tss)-1),
                                         ignore_index=True)
            run_id = sweep.runs[i].id
            run_config = sweep.runs[i].config
            run_config = {k: run_config[k] for k in
                          run_config.keys() & {'layers', 'hidden_units',
                          'batch_size',
                                               'learning_rate', 'seed',
                                               'n_features'}}
            run_config.update({'id': run_id})
            run_config = pd.DataFrame(run_config, index=range(1))
            run_config = run_config.append([run_config] * (len(val_tss)-1),
                                           ignore_index=True)

            # calculate moving avg and std
            moving_avg = val_tss.expanding(min_periods=1).mean()
            moving_std = val_tss.expanding(min_periods=1).std()
            # window averages
            moving_avg_w = val_tss.rolling(window=w).mean()
            moving_std_w = val_tss.rolling(window=w).std()
            moving_avg_g = pd.Series(np.gradient(moving_avg_w))
            moving_std_g = pd.Series(np.gradient(moving_std_w))
            moving_std_w = moving_std_w.fillna(0)
            moving_avg_w = moving_avg_w.fillna(0)
            moving_std_g = moving_std_g.fillna(0)
            moving_avg_g = moving_avg_g.fillna(0)
            moving_std = moving_std.fillna(0)
            moving_avg = moving_avg.fillna(0)
            moving_avg.name = 'moving_avg'
            moving_avg_w.name = 'moving_avg_w'
            moving_avg_g.name = 'moving_avg_g'
            moving_std.name = 'moving_std'
            moving_std_w.name = 'moving_std_w'
            moving_std_g.name = 'moving_std_g'
            test_tss.name = 'Test_TSS'
            val_tss = val_tss.to_frame()

            val_tss['epoch'] = np.arange(len(val_tss))

            # create dataframe from the run
            run_data = pd.concat([run_config, val_tss, test_tss, moving_avg,
                                  moving_std,
                                  moving_avg_w, moving_std_w, moving_avg_g,
                                  moving_std_g],
                                 axis=1)

            # append run to other runs
            all_runs = pd.concat([all_runs, run_data], axis=0, sort=False)
            print(i)

        # dataframe to csv
        all_runs.to_csv(filename)


def plot_val_std(layers, hidden_units, batch_size, learning_rate, seed,
                 avg=False):
    # select run values
    if avg:
        run_data = all_runs[(all_runs['layers'] == layers) & (
                    all_runs['hidden_units'] == hidden_units) & (all_runs[
                                                                     'batch_size'] == batch_size) & (
                                        all_runs[
                                            'learning_rate'] == learning_rate)]
        savefile = 'std_avg_{}_{}_{}_{:.0e}_{}'.format(layers, hidden_units,
                                                       batch_size,
                                                       learning_rate, seed)
    else:
        run_data = all_runs[(all_runs['layers'] == layers) & (
                    all_runs['hidden_units'] == hidden_units) & (all_runs[
                                                                     'batch_size'] == batch_size) & (
                                        all_runs[
                                            'learning_rate'] ==
                                        learning_rate) & (
                                        all_runs['seed'] == seed)]
        savefile = 'std_{}_{}_{}_{:.0e}_{}'.format(layers, hidden_units,
                                                   batch_size, learning_rate,
                                                   seed)

    # get best validation epoch and test score at that point
    test_tss = run_data.groupby('seed').apply(lambda x: x['Test_TSS'].unique())
    best_epoch_idx = run_data.groupby('seed').apply(
        lambda x: x['Validation_TSS'].idxmax())
    best_epoch = run_data['epoch'][best_epoch_idx]
    std_at_epoch = run_data['moving_std_w'][best_epoch_idx]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    color = 'tab:blue'
    ax1.set_title('Validation TSS and standard deviations', fontsize=16)
    ax1.set_xlabel('Epoch', fontsize=16)
    ax1.set_ylabel('Validation TSS', fontsize=16, color=color)
    ax1.set_ylim(0, 1)
    plt.scatter(best_epoch.to_numpy(), test_tss.to_numpy(), c='g')
    test_str = ''
    for i in range(len(test_tss)):
        test_str = test_str + '{}. {} ({})\n'.format(i, test_tss.iloc[i],
                                                     test_tss.index[i])
    plt.legend(['[Test TSS] (seed) \n' + test_str], loc=1)
    for i in range(len(best_epoch)):
        ax1.text(best_epoch.iloc[i] + 1, test_tss.iloc[i], '{}'.format(i),
                 c='g', size=10)

    ax2 = sns.lineplot(x='epoch', y='Validation_TSS', color=color,
                       data=run_data)

    ax1.tick_params(axis='y')
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Moving STD Gradient', fontsize=16, color=color)
    ax2.set_ylim(None, 0.3)
    ax2 = sns.lineplot(x='epoch', y='moving_std_w', color=color, data=run_data)
    ax2.tick_params(axis='y', color=color)
    plt.legend(['layers:{} \nnodes:{} \nbatch_size:{} \nlearning_rate:{} \n'
                'seed:{}'
                ''.format(layers, hidden_units, batch_size, learning_rate,
                          seed)], loc=4)
    plt.savefig(pathname + savefile + '.png', bbox_inches='tight')
    plt.show()


def check_network(layers, hidden_units, batch_size, learning_rate, seed,
                  hp_list=pd.DataFrame()):
    # is network above 0.79 val tss
    # is moving_std_w < 0.05
    # if true, return/flag the network
    column_names = ['layers', 'hidden_units', 'batch_size', 'learning_rate',
                    'seed', 'Test_TSS']

    run_data = all_runs[(all_runs['layers'] == layers)
                        & (all_runs['hidden_units'] == hidden_units)
                        & (all_runs['batch_size'] == batch_size)
                        & (all_runs['learning_rate'] == learning_rate)
                        & (all_runs['seed'] == seed)]

    max_val_tss = run_data['Validation_TSS'].max()
    max_val_idx = run_data['Validation_TSS'].idxmax()
    max_val_epoch = run_data['epoch'][max_val_idx]
    max_moving_std_w = run_data['moving_std_w'][max_val_idx]

    if (max_val_tss > 0.79) and (max_moving_std_w < 0.02):
        flag = True
        # add hp to dataframe

        hps = '{}_{}_{}_{:.0e}_{}'.format(layers, hidden_units, batch_size,
                                          learning_rate, seed)
        hp_list.loc[len(hp_list)] = [layers, hidden_units, batch_size,
                                     learning_rate, seed,
                                     run_data['Test_TSS'][run_data.index[0]]]
    else:
        flag = False

    return flag, hp_list


def count_valid_network(hp_list):
    # if 3 out of 5 seeds are valid
    # then return those HPs
    counted_nets = hp_list.groupby(['layers', 'hidden_units',
                                         'batch_size',
                     'learning_rate']).size().reset_index().rename(
        columns={0: 'count'})
    return counted_nets


def final_model_scores(hps, hp_list):
    # get the test scores of the chosen HP
    final_scores = pd.DataFrame()
    for i in range(len(hps)):
        row_entry = hps.iloc[i]
        layers = row_entry['layers']
        hidden_units = row_entry['hidden_units']
        batch_size = row_entry['batch_size']
        learning_rate = row_entry['learning_rate']
        all_entries = hp_list[(hp_list['layers'] == layers)
                              & (hp_list['hidden_units'] == hidden_units)
                              & (hp_list['batch_size'] == batch_size)
                              & (hp_list['learning_rate'] == learning_rate)]
        final_scores = pd.concat([final_scores, all_entries])
    return final_scores


def get_hp_from_id(id):
    run = all_runs[all_runs['id'] == id]
    fidx = run.index[0]
    run_layers = run['layers'][fidx]
    run_hidden_units = run['hidden_units'][fidx]
    run_batch_size = run['batch_size'][fidx]
    run_learning_rate = run['learning_rate'][fidx]
    run_seed = run['seed'][fidx]

    return run_layers, run_hidden_units, run_batch_size, run_learning_rate, \
           run_seed


''' 
Get best nets
'''

id_list = all_runs['id'].unique()
column_names = ['layers', 'hidden_units', 'batch_size', 'learning_rate',
                'seed', 'Test_TSS']
hp_list = pd.DataFrame(columns=column_names)
for id in id_list:
    layers, hidden_units, batch_size, learning_rate, seed = get_hp_from_id(id)
    flag, hp_list = check_network(layers, hidden_units, batch_size,
                                  learning_rate, seed, hp_list)
    bad_runs = hp_list[hp_list['Test_TSS'] < 0.85]
    good_runs = hp_list[hp_list['Test_TSS'] > 0.85]

counted_valid_hps = count_valid_network(hp_list)
final_possible_hps = counted_valid_hps[counted_valid_hps['count'] >= 3]
# Check test performance of final networks
final_net_scores = final_model_scores(final_possible_hps, hp_list)

# print(tabulate(final_possible_hps, headers="keys", tablefmt="github",
#                floatfmt=('', ".0f", ".0f", '.0f', '.0e', '.0f')
#                ))


'''
plot val moving std
'''

# layers = 2
# hidden_units = 100
# batch_size = 8192
# learning_rate = 1e-3
# seeds = [335, 49, 124, 15, 273]
# for seed in seeds:
#     plot_val_std(layers, hidden_units, batch_size, learning_rate, seed,
#                  avg=False)
# plot_val_std(layers, hidden_units, batch_size, learning_rate, seed,
#                  avg=True)
