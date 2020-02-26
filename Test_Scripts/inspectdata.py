import pandas as pd
import torch
import re
import main_TCN_Liu
from tabulate import tabulate
import seaborn as sns

filepath = './Data/Liu/' + args.flare_label + '/'
df_train = pd.read_csv(filepath + 'normalized_training.csv')
df_val = pd.read_csv(filepath + 'normalized_validation.csv')
df_test = pd.read_csv(filepath + 'normalized_testing.csv')
df = pd.concat([df_train, df_val, df_test], axis=0)

'''
Basic Stats
'''
df.dropna(inplace=True)
perc = [.20, .40, .60, .80]
include = ['object', 'float', 'int']
desc = df.describe(include=include)
# print(tabulate(desc, headers="keys", tablefmt="github"))  # latex too

# get x flares
m5_flares = df[df['flare'].str.match('M5') |
               df['flare'].str.match('M6') |
               df['flare'].str.match('M7') |
               df['flare'].str.match('M8') |
               df['flare'].str.match('M9') |
               df['flare'].str.match('X')]
m5_flared_NOAA = m5_flares['NOAA'].unique()
x_flares_data = df[df['NOAA'].isin(m5_flared_NOAA)]

# get average duration of sunspot
samples_per_AR = df['NOAA'].value_counts()
samples_per_AR.mean()
samples_per_AR.std()
# / (np.sqrt(len(samples_per_AR)))

# flares/spots per year
df['date'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%dT%H:%M:%S.%fZ')
yearly_sunspots_AR = df.groupby(df['date'].dt.year).agg(
    {'NOAA': pd.Series.nunique})
print(tabulate(yearly_sunspots_AR, headers="keys", tablefmt="github"))
df = df.drop(['date'], axis=1)

'''
Seaborn data
'''
# todo note that this is for flares that actually erupted eventually
snsdata = x_flares_data.drop(['flare', 'timestamp', 'NOAA', 'HARP'], axis=1)
# sns.pairplot(snsdata, diag_kind='kde', plot_kws={'alpha': .2})
sns_plot = sns.pairplot(snsdata, hue='label', kind='reg')
sns_plot.savefig("saved/figures/snsplot_flaredARs_reg.png")
# sns_plot.fig.show()

box_plot = sns.catplot(x='label', y='TOTUSJH', data=snsdata, kind='box')
box_plot.fig.show()

# no flares
not_flare = df[~df['NOAA'].isin(m5_flared_NOAA)]
snsdata_no = not_flare.drop(['flare', 'timestamp', 'NOAA', 'HARP'], axis=1)
sns_no_plot = sns.pairplot(snsdata_no.iloc[0:8021, :])
sns_no_plot.savefig("saved/figures/snsplot_noflares.png")

snsdata = snsdata.drop(['label'], axis=1)
snsdata_no = snsdata_no.drop(['label'], axis=1)
snsdata['flared'] = 'Positive'
snsdata_no['flared'] = 'Negative'

snsdata_all = pd.concat([snsdata, snsdata_no.iloc[0:8021, :]])
sns_all_plot = sns.pairplot(snsdata_all, hue='flared')
sns_all_plot.savefig("saved/figures/snsplot_flarevsno.png")

'''
Infer values
'''
x_flares_idx = df.index[df['NOAA'].isin(m5_flared_NOAA)].tolist()
test_sample_x = test_loader.dataset.data[x_flares_idx].to(device)
inferred = model(test_sample_x)
_, predicted = torch.max(inferred.data, 1)
df_inf = pd.DataFrame(inferred.cpu().detach().numpy(), columns=list('01'))
df_pred = pd.DataFrame(predicted.cpu().detach().numpy(), columns=list('p'))
concattable = pd.concat(
    [df_pred.reset_index(drop=True), df_inf.reset_index(drop=True),
     x_flares_data.reset_index(drop=True)], sort=False, axis=1)

'''
Tensorboard
'''
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/test')
# get some random training images
dataiter = iter(train_loader)
data, labels = dataiter.next()
# inspect model
writer.add_graph(model, data.to(device))
writer.close()  # tensorboard --logdir=runs
