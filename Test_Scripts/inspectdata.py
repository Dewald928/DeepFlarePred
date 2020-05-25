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

drop_path = os.path.expanduser(
    '~/Dropbox/_Meesters/figures/features_inspect/')
filepath = './Data/Liu/' + 'M5' + '/'
df_train = pd.read_csv(filepath + 'normalized_training.csv')
df_val = pd.read_csv(filepath + 'normalized_validation.csv')
df_test = pd.read_csv(filepath + 'normalized_testing.csv')
df = pd.concat([df_train, df_val, df_test], axis=0)
df = df.sort_values(by=['NOAA', 'timestamp'])

a = df[df.duplicated(subset=['timestamp', 'NOAA'], keep=False)]

'''
Basic Stats
'''
df.dropna(inplace=True)
include = ['object', 'float', 'int']
desc = df.describe(include=include)
# print(tabulate(desc, headers="keys", tablefmt="github"))  # latex too

# get >m5 flares
b_flares = df[df['flare'].str.contains('B')]
c_flares = df[df['flare'].str.contains('C')]
m_flares = df[df['flare'].str.contains('M')]
x_flares = df[df['flare'].str.contains('X')]
b_flares_NOAA = b_flares['NOAA'].unique()
c_flares_NOAA = c_flares['NOAA'].unique()
m_flares_NOAA = m_flares['NOAA'].unique()
x_flares_NOAA = x_flares['NOAA'].unique()
b_flares_data = df[df['NOAA'].isin(b_flares_NOAA)]
c_flares_data = df[df['NOAA'].isin(c_flares_NOAA)]
m_flares_data = df[df['NOAA'].isin(m_flares_NOAA)]
x_flares_data = df[df['NOAA'].isin(x_flares_NOAA)]

m5_flares = df[df['label'].str.match('Positive')]
m5_flared_NOAA = m5_flares['NOAA'].unique()
m5_flares_data = df[df['NOAA'].isin(m5_flared_NOAA)]

# get average duration of sunspot
# samples_per_AR = df['NOAA'].value_counts()
samples_per_AR = b_flares_data['NOAA'].value_counts()
samples_per_AR.mean() /24
samples_per_AR.std() /24
samples_per_AR.max()


# flares/spots per year
df['date'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%dT%H:%M:%S.%fZ')
yearly_sunspots_AR = df.groupby(df['date'].dt.year).agg(
    {'NOAA': pd.Series.nunique})
print(tabulate(yearly_sunspots_AR, headers="keys", tablefmt="github"))
df = df.drop(['date'], axis=1)

# todo flares count per AR
unique_flares = df.groupby('NOAA')['flare'].agg([
    pd.Series.unique]).reset_index()
unique_flares['B'] = 0
unique_flares['C'] = 0
unique_flares['M'] = 0
unique_flares['X'] = 0
for index, value in unique_flares['unique'].items():
    s = pd.Series(value)
    # print(s)
    b_count = s.str.contains('B').sum()
    c_count = s.str.contains('C').sum()
    m_count = s.str.contains('M').sum()
    x_count = s.str.contains('X').sum()
    unique_flares['B'][index] = b_count
    unique_flares['C'][index] = c_count
    unique_flares['M'][index] = m_count
    unique_flares['X'][index] = x_count
unique_flares = unique_flares.drop(columns=['unique'])
print(tabulate(unique_flares[unique_flares['X'] >= 1], headers="keys",
               tablefmt="github", showindex=False))


'''
Feature Pairplots
'''
def corr(x, y, **kwargs):
    # Function to calculate correlation coefficient between two arrays
    # Calculate the value
    coef = np.corrcoef(x, y)[0][1]
    # Make the label
    label = r'$\rho$ = ' + str(round(coef, 2))
    # Add the label to the plot
    ax = plt.gca()
    ax.annotate(label, xy=(0.2, 0.95), size=20, xycoords=ax.transAxes)
    if coef > 0.9:
        # get axis
        xlabel = ax.xaxis.get_label_text()
        ylabel = ax.yaxis.get_label_text()
        labs = pd.DataFrame([xlabel, ylabel])
        global corr_features_df
        corr_features_df = pd.concat([corr_features_df, labs.T])


# colour palette
colours = ["#3498db", "r"]
sns.set_palette(colours)
# sns.palplot(sns.color_palette(colours))


# ARs that eventually erupt large (24 hour ahead)
snsdata = m5_flares_data.drop(['flare', 'timestamp', 'NOAA', 'HARP'], axis=1)
sns_plot = sns.pairplot(snsdata, hue='label',
                        hue_order=['Negative', 'Positive'],
                        # vars=['Cdec', 'Chis1d', 'Edec'],
                        plot_kws={'alpha': 0.6, 's': 80, 'edgecolor': 'w'},
                        height=4
                        )
sns_plot.savefig(drop_path + "feature_pairplot_flaredARs.png")
# sns_plot.fig.show()

box_plot = sns.catplot(x='label', y='TOTUSJH', data=snsdata, kind='box')
box_plot.fig.show()


# ARs that flare >M5 vs ARs that never erupt >M5
not_flare = df[~df['NOAA'].isin(m5_flared_NOAA)]
snsdata_no = not_flare.drop(['flare', 'timestamp', 'NOAA', 'HARP'], axis=1)
# sns_no_plot = sns.pairplot(snsdata_no.iloc[0:8021, :])
# sns_no_plot.savefig("saved/figures/snsplot_noflares.png")

snsdata = snsdata.drop(['label'], axis=1)
snsdata_no = snsdata_no.drop(['label'], axis=1)
snsdata['flared'] = 'Positive'
snsdata_no['flared'] = 'Negative'

snsdata_all = pd.concat([snsdata, snsdata_no.iloc[0:len(snsdata), :]])
sns_all_plot = sns.pairplot(snsdata_all, hue='flared',
                            hue_order=['Negative', 'Positive'],
                            # vars=['Cdec', 'Chis1d', 'Edec', 'EPSY'],
                            plot_kws={'alpha': 0.6, 's': 80, 'edgecolor': 'w'},
                            height=4, corner=True)
sns_all_plot.savefig(drop_path + "feature_pairplot_flarevsno.png")

# todo log skewed features


# Pair Grid
cmap = sns.cubehelix_palette(as_cmap=True, dark=0, light=1, reverse=True)
g = sns.PairGrid(snsdata, hue='flared', hue_order=['Negative', 'Positive'],
                 vars=['Cdec', 'Chis1d', 'Edec'], height=4, **{'diag_sharey': False})
g.map_upper(plt.scatter, edgecolor="white", alpha=0.6, s=80)
g.map_lower(sns.kdeplot)
g.map_diag(sns.kdeplot, **{'shade': True})
g.add_legend(loc='upper center')
g.savefig(drop_path + "feature_pairplot_flaredAR_mixed.png")
plt.show()


# Check correlation
corr_plot = sns.PairGrid(snsdata)
corr_plot.map_upper(plt.scatter, edgecolor="white")
corr_plot.map_lower(sns.kdeplot)
corr_plot.map_diag(sns.kdeplot)
corr_plot.add_legend(loc='upper right')

xlabels, ylabels = [], []

for ax in corr_plot.axes[-1, :]:
    xlabel = ax.xaxis.get_label_text()
    xlabels.append(xlabel)
for ax in corr_plot.axes[:, 0]:
    ylabel = ax.yaxis.get_label_text()
    ylabels.append(ylabel)

for i in range(len(xlabels)):
    for j in range(len(ylabels)):
        corr_plot.axes[j, i].xaxis.set_label_text(xlabels[i])
        corr_plot.axes[j, i].yaxis.set_label_text(ylabels[j])

corr_features_df = pd.DataFrame()
corr_plot.map_lower(corr)
corr_plot.savefig(drop_path + "feature_pairplot_correlation.png")
# plt.show()

print(tabulate(corr_features_df.T, headers="keys",
               tablefmt="github", showindex=False))


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
     m5_flares_data.reset_index(drop=True)], sort=False, axis=1)
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
