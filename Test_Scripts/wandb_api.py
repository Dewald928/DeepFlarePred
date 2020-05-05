import numpy as np
import pandas as pd
import wandb
import os.path
api = wandb.Api()

# Get runs from a specific sweep
sweep0 = api.sweep("dewald123/liu_pytorch_MLP/xris1c8z")
sweep1 = api.sweep("dewald123/liu_pytorch_MLP/0iihk32s")
sweep2 = api.sweep("dewald123/liu_pytorch_MLP/1c6ig5mc")
sweep = api.sweep("dewald123/liu_pytorch_MLP/u8kq0stw")
sweeps = [sweep, sweep1, sweep2]

# Get all runs validation_TSS,
# runs_val = pd.DataFrame()
# for i in range(len(sweep.runs)):
#     runs_val = pd.concat([runs_val, sweep.runs[i].history().Validation_TSS],
#                          axis=1, sort=False)
#     print(i)
#
# runs_val.to_csv('val_tss')

# todo name arch and hp

# if the file is already written to, read it
runs_avg = pd.read_csv('runs_avg') if os.path.isfile('runs_avg') is True \
    else pd.DataFrame()
runs_std = pd.read_csv('runs_std') if os.path.isfile('runs_std') is True \
    else pd.DataFrame()
runs_tss = pd.read_csv('runs_tss') if os.path.isfile('runs_tss') is True \
    else pd.DataFrame()
runs_cfg = pd.read_csv('runs_cfg') if os.path.isfile('runs_cfg') is True \
    else pd.DataFrame()

for i in range(len(sweep.runs)):
    # wandb.init(project='liu_pytorch_MLP', entity='dewald123',
    #            name=sweep.runs[i].name, id=sweep.runs[i].id, resume=True,
    #            config=sweep.runs[i].config)
    # sweep.runs[i].update()
    # get val tss for run
    history = sweep.runs[i].history(samples=1000)
    val_tss = history["Validation_TSS"]
    run_config = sweep.runs[i].config
    run_id = sweep.runs[i].id
    run_config.update({'id': run_id})
    val_tss.name = run_id
    # calculate moving avg and std
    moving_avg = val_tss.expanding(min_periods=1).mean()
    moving_std = val_tss.expanding(min_periods=1).std()
    moving_std = moving_std.fillna(0)
    moving_avg = moving_avg.fillna(0)

    # create dataframes
    runs_tss = pd.concat([runs_tss, val_tss], axis=1, sort=False)
    runs_avg = pd.concat([runs_avg, moving_avg], axis=1, sort=False)
    runs_std = pd.concat([runs_std, moving_std], axis=1, sort=False)
    if (i == 0) and (os.path.isfile('runs_cfg') is False):
        runs_cfg = pd.DataFrame.from_dict(run_config, orient='columns')
    else:
        runs_cfg = pd.concat(
            [runs_cfg, pd.DataFrame.from_dict(run_config, orient='index').T],
            axis=0, sort=False)
    print(i)

# dataframe to csv
runs_tss.to_csv('runs_tss')
runs_avg.to_csv('runs_avg')
runs_std.to_csv('runs_std')
runs_cfg.to_csv('runs_cfg')

    # log to original run
    # for s in range(len(val_tss)-1):
    #     wandb.log({'moving_avg_tss': moving_avg.to_numpy()[s]}, step=s)
    #     wandb.log({'moving_std_tss': moving_std.to_numpy()[s]}, step=s)
    # sweep.runs[i].update()






