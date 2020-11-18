import wandb
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os

api = wandb.Api()

# list of runs ids
dropbox = os.path.expanduser(
    '~/Dropbox/_Meesters/figures/training_curves/')
# hps = ['batch_size']
path = 'dewald123/liu_pytorch_MLP/'
id_dict = {0: [], 1: []}
# id_dict = {0: []}
id_dict[0] = ['nzgx1hv5', '16z66cnv', 'bugqi272']
id_dict[1] = ['u5ewgdty', 'g7wjz699', '99nvkoax']
runs_dict = {0: pd.DataFrame(), 1: pd.DataFrame()}
# runs_dict = {0: pd.DataFrame()}
colors = ['g', 'm']

fig = plt.figure(figsize=(10,5))

# Project is specified by <entity/project-name>
for i in range(len(id_dict)):
    for s in range(len(id_dict[i])):
        run = api.run(f"{path}{id_dict[i][s]}")
        if run.state == "finished":
            df = run.history().loc[:, ['Validation_TSS', 'Training_TSS', 'Test_TSS_curve']].dropna()
            runs_dict[i] = pd.concat([runs_dict[i], df])

    # plot curves
    # sns.lineplot(data=runs_dict[i], x=runs_dict[i].index, y="Training_TSS", ci=68)
    sns.lineplot(data=runs_dict[i], x=runs_dict[i].index, y="Validation_TSS", ci=68, color=colors[i])
    # sns.lineplot(data=runs_dict[i], x=runs_dict[i].index, y="Test_TSS_curve", ci=68)

# save images
plt.xlim(xmin=0, xmax=450)
# plt.legend(['Batch size: 65 536', 'Batch size: 1 024'])
plt.legend(['Scaling strategy: z_train', 'Scaling strategy: z_minmax_train'], loc=4)
plt.xlabel('Epoch')
plt.ylabel('TSS')
plt.title('Training curves. MLP (1_100). Validation TSS.')

plt.tight_layout()
plt.savefig(dropbox + 'TC_scaler_comp.pdf')
plt.show()