import pandas as pd
import torch
import re
import main_TCN_Liu
from tabulate import tabulate
import seaborn as sns
import scipy
import matplotlib.pyplot as plt

filepath = './Data/Liu/' + 'M5' + '/'
df_train = pd.read_csv(filepath + 'normalized_training.csv')
df_val = pd.read_csv(filepath + 'normalized_validation.csv')
df_test = pd.read_csv(filepath + 'normalized_testing.csv')
df = pd.concat([df_train, df_val, df_test], axis=0)
df = df.sort_values(by=['NOAA', 'timestamp'])

a = df[df.duplicated(subset=['timestamp', 'NOAA'], keep=False)]

'''
flare to flux
'''
def flare_to_flux(flare_class):
    flux = 1e-7
    if flare_class == 'N':
        flux = 1e-7
    else:
        class_label = flare_class[0]
        class_mag = float(flare_class[1:])
        if class_label=='B':
            flux = 1e-7 * class_mag
        if class_label=='C':
            flux = 1e-6 * class_mag
        if class_label=='M':
            flux = 1e-5 * class_mag
        if class_label=='X':
            flux = 1e-4 * class_mag
    return flux

flux = df['flare'].apply(flare_to_flux)
flux.name = 'flux'
df_flux = pd.concat([flux, df], axis=1)


'''
Basic Stats
'''
df.dropna(inplace=True)
perc = [.20, .40, .60, .80]
include = ['object', 'float', 'int']
desc = df.describe(include=include)
# print(tabulate(desc, headers="keys", tablefmt="github"))  # latex too

# get x flares
m5_flares = df[df['label'].str.match('Positive')]
m5_flared_NOAA = m5_flares['NOAA'].unique()
x_flares_data = df[df['NOAA'].isin(m5_flared_NOAA)]

# add flux
flux = x_flares_data['flare'].apply(flare_to_flux)
flux.name = 'flux'
df_flux = pd.concat([flux, x_flares_data], axis=1)

fig, ax = plt.subplots()
ax.set(yscale='log')
sns.lineplot(x='timestamp', y='flux', hue='NOAA',
             data=df_flux,
             ax=ax, sort=False)
plt.show()

# get average duration of sunspot
samples_per_AR = df['NOAA'].value_counts()
# samples_per_AR = x_flares_data.value_counts()
samples_per_AR.mean()
samples_per_AR.std()
samples_per_AR.max()
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
sns_plot = sns.pairplot(snsdata, hue='label',
                        hue_order=['Negative', 'Positive'])
sns_plot.savefig("saved/figures/snsplot_flaredARs_order.png")
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

# sns testing
df_iris = sns.load_dataset('iris')
cmap = sns.cubehelix_palette(as_cmap=True, dark=0, light=1, reverse=True)
g = sns.PairGrid(snsdata, hue='label', hue_order=['Negative', 'Positive'])
g.map_upper(plt.scatter)
g.map_lower(sns.kdeplot)
g.map_diag(sns.kdeplot)
g.savefig("saved/figures/snsplot_flaredARs_mixed.png")
# plt.show()

'''
Infer values
'''
x_flares_idx = df.index[df['NOAA'].isin(m5_flared_NOAA)].tolist()
test_sample_x = test_loader.dataset.data[x_flares_idx].to(device)
inferred = model(test_sample_x)
_, predicted = torch.max(inferred.data, 1)
df_inf = pd.DataFrame(inferred.cpu().detach().numpy()[:, 1], columns=list('1'))
df_pred = pd.DataFrame(predicted.cpu().detach().numpy(), columns=list('p'))
concattable = pd.concat(
    [df_pred.reset_index(drop=True), df_inf.reset_index(drop=True),
     x_flares_data.reset_index(drop=True)], sort=False, axis=1)
ARs_classified = concattable[(concattable['p'] == 1) & (concattable['label']
                                                        == 'Positive')]
ARs_classified_NOAA = ARs_classified['NOAA'].unique()
numberofAR = len(m5_flared_NOAA) - len(ARs_classified_NOAA)


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
