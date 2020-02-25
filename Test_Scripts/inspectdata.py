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
# print(tabulate(desc, headers="keys", tablefmt="github"))  # github for markdown

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
samples_per_AR = x_flares_data['NOAA'].value_counts()
samples_per_AR.mean()

# get good sunspots (that move from left to right)

# flares/spots per year

'''
Seaborn data
'''
snsdata = x_flares_data.drop(['flare', 'timestamp', 'NOAA', 'HARP'], axis=1)
# sns.pairplot(snsdata, diag_kind='kde', plot_kws={'alpha': .2})
sns.pairplot(snsdata.iloc[0:3, 0:4], diag_kind='kde', plot_kws={'alpha': .2})
plt.show()

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
