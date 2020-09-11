import numpy as np
import pandas as pd
import wandb
import os.path
import os
import seaborn as sns
import matplotlib.pyplot as plt
from tabulate import tabulate
from matplotlib.legend import Legend

api = wandb.Api()
# specify HPs
model_type = 'MLP'  # 'TCN 'or 'MLP'
if model_type == 'MLP':
    HP_list = ['dataset', 'layers', 'hidden_units', 'batch_size',
               'learning_rate', 'seed']
    # HP_list = ['layers', 'hidden_units', 'batch_size', 'learning_rate',
    # 'seed']
    HP_groupby = ['dataset', 'layers', 'hidden_units', 'batch_size',
                  'learning_rate']  # HP_groupby = ['layers',
    # 'hidden_units', 'batch_size', 'learning_rate']
elif model_type == 'TCN':
    HP_list = ['levels', 'ksize', 'seq_len', 'nhid', 'dropout', 'batch_size',
               'learning_rate', 'weight_decay', 'seed']
    HP_groupby = ['levels', 'ksize', 'seq_len', 'nhid', 'dropout',
                  'batch_size', 'learning_rate', 'weight_decay']

num_seeds = 3
filename = 'MLP_40f_2.csv'
pathname = os.path.expanduser(
    '~/Dropbox/_Meesters/figures/moving_std_val_tss/MLP/40feat/')
all_runs = pd.read_csv(pathname + filename) if os.path.isfile(
    pathname + filename) is True else pd.DataFrame()


def get_wandb_runs():
    # Get runs from a specific sweep
    sweep0 = api.runs(path="dewald123/liu_pytorch_MLP",
                      filters={'sweep': "919m8v0h"}, per_page=1200)
    sweep0 = api.runs(path="dewald123/liu_pytorch_MLP",
                      filters={'sweep': "sapcuf56"}, per_page=1200)
    # sweep1 = api.sweep("dewald123/liu_pytorch_MLP/2yk81dhq")
    # sweep2 = api.sweep("dewald123/liu_pytorch_MLP/1c6ig5mc")
    # sweep3 = api.sweep("dewald123/liu_pytorch_MLP/grw0rrpp")

    sweeps = [sweep0, sweep1]

    for sweep in sweeps:
        # if the file is already written to, read it
        all_runs = pd.read_csv(pathname + filename) if os.path.isfile(
            pathname + filename) is True else pd.DataFrame()

        for i in range(len(sweep)):
            w = 10
            run_data = pd.DataFrame()
            # get val tss for run
            if sweep[i].state == 'finished':
                history = sweep[i].history(samples=2000)
            else:
                continue
            val_tss = history["Validation_TSS"].dropna().reset_index(drop=True)
            try:
                test_tss = history["Test_TSS"].dropna().reset_index(drop=True)
            except:
                test_tss = pd.Series(
                    history['Test_TSS_curve'][val_tss.idxmax()])
            test_tss = test_tss.append([test_tss] * (len(val_tss) - 1),
                                       ignore_index=True)
            best_val_epoch = history[
                'Best_Validation_epoch'].dropna().reset_index(drop=True)
            best_val_epoch = best_val_epoch.append(
                [best_val_epoch] * (len(val_tss) - 1), ignore_index=True)
            run_id = sweep[i].id
            run_config = sweep[i].config
            run_config = {k: run_config[k] for k in
                          run_config.keys() & set(HP_list)}
            run_config.update({'id': run_id})
            run_config = pd.DataFrame(run_config, index=range(1))
            run_config = run_config.append([run_config] * (len(val_tss) - 1),
                                           ignore_index=True)

            # calculate moving avg and std
            moving_avg = val_tss.expanding(min_periods=1).mean()
            moving_std = val_tss.expanding(min_periods=1).std()
            # window averages
            moving_avg_w = val_tss.rolling(window=w).mean()
            moving_std_w = val_tss.rolling(window=w).std()
            moving_avg_w = moving_avg_w.fillna(val_tss.expanding().mean())
            moving_std_w = moving_std_w.fillna(val_tss.expanding().std())
            moving_avg_g = pd.Series(np.gradient(moving_avg_w))
            moving_std_g = pd.Series(np.gradient(moving_std_w))
            moving_avg_w = moving_avg_w.fillna(0)
            moving_std_w = moving_std_w.fillna(0)
            moving_avg_g = moving_avg_g.fillna(0)
            moving_std_g = moving_std_g.fillna(0)
            moving_avg = moving_avg.fillna(0)
            moving_std = moving_std.fillna(0)
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
            run_data = pd.concat(
                [run_config, val_tss, best_val_epoch, test_tss, moving_avg,
                 moving_std, moving_avg_w, moving_std_w, moving_avg_g,
                 moving_std_g], axis=1)

            # append run to other runs
            all_runs = pd.concat([all_runs, run_data], axis=0, sort=False)
            print(i)

        # dataframe to csv
        all_runs.to_csv(pathname + filename, index=False)


def plot_val_std_MLP(layers, hidden_units, batch_size, learning_rate, seed,
                     avg=False):
    # select run values
    if avg:
        run_data = all_runs[(all_runs['layers'] == layers) & (
                all_runs['hidden_units'] == hidden_units) & (all_runs[
                                                                 'batch_size'] == batch_size) & (
                                    all_runs['learning_rate'].round(
                                        4) == learning_rate)]
        savefile = 'std_avg_{}_{}_{}_{:.0e}_{}'.format(layers, hidden_units,
                                                       batch_size,
                                                       learning_rate, seed)
    else:
        run_data = all_runs[(all_runs['layers'] == layers) & (
                all_runs['hidden_units'] == hidden_units) & (all_runs[
                                                                 'batch_size'] == batch_size) & (
                                    all_runs['learning_rate'].round(
                                        4) == learning_rate) & (
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
    best_val_tss = run_data.groupby('seed').apply(
        lambda x: x['Validation_TSS'][best_epoch_idx]).dropna(how='any',
                                                              axis=1)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    color = 'tab:blue'
    ax1.set_title('Validation TSS and standard deviations', fontsize=16)
    ax1.set_xlabel('Epoch', fontsize=16)
    ax1.set_ylabel('Validation TSS', fontsize=16, color=color)
    ax1.set_ylim(0, 1)
    scat1 = plt.scatter(best_epoch.to_numpy(), test_tss.to_numpy(), c='g')
    scat2 = plt.scatter(best_epoch.to_numpy(), best_val_tss.to_numpy(), c='b')
    test_str = []
    for i in range(len(test_tss)):
        test_str = np.append(test_str,
                             '{}. {:.4f} ({})'.format(i, test_tss.iloc[i][0],
                                                      test_tss.index[i]))
    plt.legend(test_str, title='[Test TSS] (seed)', loc=1)
    best_val_tss_str = []
    for i in range(len(best_val_tss)):
        best_val_tss_str = np.append(best_val_tss_str, '{}. {:.4f} ({'
                                                       '})'.format(i,
                                                                   best_val_tss.iloc[
                                                                       i].values[
                                                                       0],
                                                                   best_val_tss.index[
                                                                       i]))
    leg = Legend(ax1, [scat2], [best_val_tss_str] if len(
        best_val_tss_str) == 1 else best_val_tss_str, title='[Validation '
                                                            'TSS] ('
                                                            'seed)', loc=9)
    ax1.add_artist(leg)
    # plt.legend(['[Validation TSS] (seed) \n' + best_val_tss_str], loc=2)
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


def plot_val_std(run_hp, avg=False):
    # get run id from run_hp
    id = get_id_from_hp(run_hp).iloc[0]
    # select run values
    if avg:
        run_data = all_runs[(all_runs['id'] == id)]
        savefile = 'std_avg_{}_{}_{}_{:.0e}_{}'.format(layers, hidden_units,
                                                       batch_size,
                                                       learning_rate, seed)
    else:
        run_data = all_runs[(all_runs['id'] == id)]
        hp_string = ''
        for i in range(run_data[HP_list].shape[1]):
            if len(str(run_data[HP_list].iloc[0][i])) < 4:
                tmp_string = f'_{run_data[HP_list].iloc[0][i]:.0f}'
            else:
                tmp_string = f'_{run_data[HP_list].iloc[0][i]:.1e}'
            hp_string = hp_string + tmp_string
        savefile = 'std{}'.format(hp_string)

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
    # plt.legend(['layers:{} \nnodes:{} \nbatch_size:{} \nlearning_rate:{} \n'
    #             'seed:{}'
    #             ''.format(layers, hidden_units, batch_size, learning_rate,
    #                       seed)], loc=4)
    plt.savefig(pathname + savefile + '.png', bbox_inches='tight')
    plt.show()


def check_network(id, val_tss_th, val_std_th, HP_list, hp_df=pd.DataFrame()):
    # is network above 0.79 val tss
    # is moving_std_w < 0.02
    # if true, return/flag the network

    run_data = all_runs[(all_runs['id'] == id)]

    max_val_tss = run_data['Validation_TSS'].max()
    max_val_idx = run_data['Validation_TSS'].idxmax()
    max_val_epoch = run_data['epoch'][max_val_idx]
    max_moving_std_w = run_data['moving_std_w'][max_val_idx]

    if (max_val_tss > val_tss_th) and (max_moving_std_w < val_std_th):
        flag = True
        # add hp to dataframe
        hp_df.loc[len(hp_df)] = run_data[HP_list].iloc[0].to_list() + [id,
                                                                       max_val_tss,
                                                                       run_data[
                                                                           'Test_TSS'].iloc[
                                                                           0]]
    else:
        flag = False
    return flag, hp_df


def count_valid_network(hp_df):
    # if 3 out of 5 seeds are valid
    # then return those HPs, unless bayesian optimization is used
    new_HP_list = list(HP_list)
    new_HP_list.remove('seed')
    counted_nets = hp_df.groupby(HP_groupby).size().reset_index(

    ).rename(columns={0: 'count'})
    return counted_nets


def final_model_scores(hps, hp_df):
    # get the test scores of the chosen HP
    final_scores = pd.DataFrame()
    all_entries = pd.merge(hp_df, hps, on=HP_groupby, how='inner')
    all_entries = all_entries.drop('count', axis=1)
    # val_se = all_entries['Validation_TSS'].sem()
    # test_se = all_entries['Validation_TSS'].sem()
    return all_entries


def get_hp_from_id(id, HP_list):
    run = all_runs[all_runs['id'] == id]
    run_hp = run[HP_list].iloc[0]
    return run_hp


def get_id_from_hp(hp_series_toget):
    hp_series_toget = pd.DataFrame(hp_series_toget).T
    possible_run = pd.merge(all_runs, hp_series_toget,
                            on=list(hp_series_toget.columns), how='inner')
    return possible_run['id']


''' 
Get wandb runs, save to csv
'''
# get_wandb_runs()

all_runs = pd.read_csv(pathname + filename) if os.path.isfile(
    pathname + filename) is True else pd.DataFrame()
all_runs = all_runs[all_runs['dataset'] == 'Liu/z_train/']
# all_runs = all_runs[all_runs['Best_Validation_epoch'] <= 40]

# get the validation tss threshold at 30% of the top values.
sorted_tss = all_runs.groupby('id')[
    'Validation_TSS'].max().reset_index().sort_values(by='Validation_TSS',
                                                      ascending=False).reset_index(
    drop=True)
ax = sns.regplot(x=sorted_tss.index, y="Validation_TSS", data=sorted_tss)
line = ax.lines[0]
tss_min = line.get_ydata()[-1]
val_tss_th = sorted_tss["Validation_TSS"].max() - (
        (sorted_tss["Validation_TSS"].max() - tss_min) * 0.5)
# val_tss_th = sorted_tss.quantile(0.8)['Validation_TSS']
# get std threshold at 70% of smallest values.
sorted_std_idx = all_runs.groupby('id')['Validation_TSS'].idxmax()
sorted_std = all_runs['moving_std_w'][
    sorted_std_idx].sort_values().reset_index(drop=True)
# val_std_th = sorted_std.quantile(0.85)
val_std_th = sorted_std.min() + ((sorted_std.max() - sorted_std.min()) * 0.05)

# plot all runs tss and mvg_std_w
plt.plot(sorted_std)
plt.plot(sorted_std.iloc[
             (sorted_std - val_std_th).abs().argsort()[:1]].index.tolist(),
         val_std_th, 'bo', label='STD Threshold')
plt.plot(sorted_tss['Validation_TSS'])
plt.plot(sorted_tss.iloc[(sorted_tss['Validation_TSS'] - val_tss_th).abs(

).argsort()[:1]].index.tolist(), val_tss_th, 'ro', label='TSS Threshold')
plt.legend()
plt.title('TSS and moving_std Threshold Selection')
plt.show()
'''
Get best nets
'''
id_list = all_runs['id'].unique()
column_names = HP_list + ['id', 'Best_Validation_TSS', 'Test_TSS']
hp_df = pd.DataFrame(columns=column_names)
for id in id_list:
    run_hp = get_hp_from_id(id, HP_list)
    flag, hp_df = check_network(id, val_tss_th, val_std_th, HP_list, hp_df)

counted_valid_hps = count_valid_network(hp_df)
final_possible_hps = counted_valid_hps[counted_valid_hps['count'] >= num_seeds]
# Check test performance of final networks
final_net_scores = final_model_scores(final_possible_hps, hp_df)

print(tabulate(counted_valid_hps, headers="keys", tablefmt="github",
               floatfmt=(".0f", ".0f", '.0f', '.0e', '.4f', '.4f'),
               showindex=False))

# get top 3 seed per arch
top3_tbl = final_net_scores.groupby(HP_groupby).apply(
    lambda x: x.nlargest(3, ['Best_Validation_TSS', 'Test_TSS'])).reset_index(
    drop=True)
# create final tables with average and standard errors
final_table = top3_tbl.groupby(HP_groupby).mean().reset_index().rename(
    columns={'Best_Validation_TSS': 'avg_val_TSS', 'Test_TSS': 'avg_test_TSS'})
# final_table = top3_tbl.groupby(HP_groupby).mean().reset_index().drop(
#     columns='seed').rename(
#     columns={'Best_Val_TSS': 'avg_val_TSS', 'Test_TSS': 'avg_test_TSS'})

hp_se = top3_tbl.groupby(HP_groupby).sem().reset_index().rename(
    columns={'Best_Validation_TSS': 'val_se', 'Test_TSS': 'test_se'})
final_table.insert(final_table.columns.get_loc("avg_val_TSS") + 1, 'val_se',
                   hp_se['val_se'])
final_table.insert(final_table.columns.get_loc("avg_test_TSS") + 1, 'test_se',
                   hp_se['test_se'])

sorted_ft = final_table.sort_values(['avg_val_TSS', 'val_se'],
                                    ascending=[False, True])

print(tabulate(sorted_ft, headers="keys", tablefmt="github", floatfmt=(
    ".0f", ".0f", '.0f', '.1e', '.4f', '.4f', '.4f', '.4f', '.4f'),
               showindex=False))

'''
Save the runs that qualified
'''
final_net_scores.to_csv(pathname+filename[:-4]+'_valid.csv', index=False)

'''
plot val moving std std
'''
layers = 2
hidden_units = 500
batch_size = 65536
learning_rate = 0.0052
# seeds = [335, 49, 124, 15, 273]
# seeds = [49]
seeds = [49, 124, 15]
for seed in seeds:
    plot_val_std_MLP(layers, hidden_units, batch_size, learning_rate, seed,
                     avg=False)
plot_val_std_MLP(layers, hidden_units, batch_size, learning_rate, seed,
                 avg=True)

'''
plot val moving std std TCN
'''
# run_hp = get_hp_from_id(final_net_scores['id'].iloc[2], HP_list)
# plot_val_std(run_hp)



