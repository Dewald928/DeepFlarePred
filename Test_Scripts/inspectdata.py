import pandas as pd
import torch
import re

# from scipy.stats import norm_gen

import main_TCN_Liu
from tabulate import tabulate
import seaborn as sns
import scipy
import matplotlib.pyplot as plt
import numpy as np
import os
from data_loader import data_loader

listofuncorrfeatures = ['TOTUSJH', 'SAVNCPP', 'ABSNJZH', 'TOTPOT', 'AREA_ACR',
                        'Cdec', 'Chis', 'Edec', 'Mhis', 'Xmax1d', 'Mdec',
                        'MEANPOT', 'R_VALUE', 'Mhis1d', 'MEANGAM', 'TOTFX',
                        'MEANJZH', 'MEANGBZ', 'TOTFZ', 'TOTFY', 'logEdec',
                        'EPSZ', 'MEANGBH', 'MEANJZD', 'Xhis1d', 'Xdec', 'Xhis',
                        'EPSX', 'EPSY', 'Bhis', 'Bdec', 'Bhis1d']  # 32

drop_path = os.path.expanduser(
    '~/Dropbox/_Meesters/figures/features_inspect/')
# filepath = './Data/Krynauw/'
# filepath = './Data/Liu_train/'
filepath = './Data/Liu/' + 'M5' + '/'
df_train = pd.read_csv(filepath + 'normalized_training.csv')
df_val = pd.read_csv(filepath + 'normalized_validation.csv')
df_test = pd.read_csv(filepath + 'normalized_testing.csv')
df = pd.concat([df_train, df_val, df_test], axis=0)
df = df.sort_values(by=['NOAA', 'timestamp'])

a = df[df.duplicated(subset=['timestamp', 'NOAA'], keep=False)]

# from sklearn.preprocessing import StandardScaler
# standardscaler = StandardScaler()
# df_norm = standardscaler.fit_transform(df.iloc[:,5:])


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

m_x_flares_data = pd.concat([m_flares_data, x_flares_data])

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



'''
Flare history histogram
'''
flares_df = pd.read_csv('./Data/GOES/all_flares_list.csv').drop\
        (columns="Unnamed: 0")
flares_df = flares_df.sort_values(by=['noaa_active_region',
                                            'peak_time']).reset_index(
    drop=True)
flares_df_prep = flares_df.copy()
flares_df_prep['goes_class'] = flares_df_prep['goes_class'].apply(lambda x:
                                                                  x[0])
flares_df_prep = flares_df_prep.pivot_table(index='noaa_active_region', columns='goes_class',
                           aggfunc={'goes_class': pd.Series.count}).fillna(0)
flares_df_prep.columns = ['B', 'C', 'M', 'X']
ar_per_class = flares_df_prep['X'].gt(0).sum()
fp0 = flares_df_prep[flares_df_prep['X'] >= 1]
fp1 = flares_df_prep[flares_df_prep['X'] < 1]
# get values before only.
flares_df_class_simplify = flares_df.copy()
# flares_df_class_simplify['goes_class'] = flares_df_class_simplify[
#     'goes_class'].apply(lambda x: x[0])
flares_from_noaa = flares_df_class_simplify[flares_df_class_simplify[
    'noaa_active_region'].isin(m5_flared_NOAA)]


HARP_NOAA_list = pd.read_csv(
    'http://jsoc.stanford.edu/doc/data/hmi/harpnum_to_noaa/all_harps_with_noaa_ars.txt', sep=' ')
listofactiveregions = list(flares_from_noaa['noaa_active_region'].unique())
new_flares_df = pd.DataFrame()
for i, noaa in enumerate(flares_from_noaa[
                           'noaa_active_region'].unique()):
    print(i)
    idx = HARP_NOAA_list[HARP_NOAA_list['NOAA_ARS'].str.contains(
        str(int(listofactiveregions[i])))]
    # if there's no HARPNUM, quit
    if idx.empty:
        print('skip: there are no matching HARPNUMs for',
              str(int(listofactiveregions[i])))
        continue
    fl_noaa_df = flares_from_noaa[flares_from_noaa[
        'noaa_active_region']==noaa]
    minimum_class_label = ['M5', 'M6', 'M7', 'M8', 'M9', 'X']
    last_idx = fl_noaa_df.index[fl_noaa_df[
        'goes_class'].str.contains('|'.join(minimum_class_label))][-1]
    new_flares_df = pd.concat([new_flares_df, fl_noaa_df.loc[:last_idx, :]])

fp0 = new_flares_df.copy()
fp0['goes_class'] = new_flares_df['goes_class'].apply(lambda x: x[0])
fp0 = fp0.pivot_table(index='noaa_active_region', columns='goes_class',
                           aggfunc={'goes_class': pd.Series.count}).fillna(0)
fp0.columns = ['B', 'C', 'M', 'X']

print(tabulate(fp0, headers="keys",
               tablefmt="github"))

fp1[['B', 'C', 'M', 'X']] = fp1[['B', 'C', 'M', 'X']].replace({0:np.nan})
fp0[['B', 'C', 'M', 'X']] = fp0[['B', 'C', 'M', 'X']].replace({0:np.nan})
flare_classes = ['B', 'C', 'M']
fps = [fp0, fp1]
fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(16,9))
for i, fp in enumerate(fps):
    # Iterate through the five airlines
    for flare_class in flare_classes:
        # Subset to the airline
        subset = fp['{}'.format(flare_class)]

        # Draw the density plot
        sns.distplot(fp.loc[:, '{}'.format(flare_class)], hist=False,
                     kde=True, rug=True,
                     kde_kws={'bw': 0.2}, label=flare_class, ax=ax[i])

    # Plot formatting
    ax[i].legend(prop={'size': 16}, title='Class')
    ax[i].set(xlabel='Number of flares per AR')
    ax[i].set(ylabel='Frequency')
    ax[0].set_title('Density Plot with Flares classes >= M5.0')
    ax[1].set_title('Density Plot with Flares classes < M5.0')
plt.savefig(drop_path + 'history_density.png')
plt.show()



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
    if coef > 0.80:
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
# sns_plot = sns.PairGrid(snsdata, hue='label', hue_order=['Negative',
#                                                           'Positive'],
#                         height=4, **{'diag_sharey': False})
# sns_plot.map_lower(sns.kdeplot)
# sns_plot.map_diag(sns.kdeplot, **{'shade': True})
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
sns_all_plot = sns.PairGrid(snsdata_all, hue='flared',
                            hue_order=['Negative', 'Positive'], height=4,
                            **{'diag_sharey': False})
sns_all_plot.map_lower(sns.kdeplot)
sns_all_plot.map_diag(sns.kdeplot, **{'shade': True})
# sns_all_plot = sns.pairplot(snsdata_all, hue='flared',
#                             hue_order=['Negative', 'Positive'],
#                             # vars=['Cdec', 'Chis1d', 'Edec', 'EPSY'],
#                             plot_kws={'alpha': 0.6, 's': 80, 'edgecolor': 'w'},
#                             height=4, corner=True)
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
# corr_plot.map_upper(plt.scatter, edgecolor="white")
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

print(tabulate(corr_features_df, headers="keys", tablefmt="github",
               showindex=False))

'''
Heatmaps of Correlation
'''
sns.set(font_scale=1.4)
# df_corr = m5_flares_data[m5_flares_data['label']=='Positive'].iloc[:,5:].corr()
# df_corr = m5_flares_data.iloc[:,5:].corr()
# df_corr = df.iloc[:, 5:].corr()
df_corr = df.loc[:,listofuncorrfeatures].corr()
df_corr[df_corr == 1] = 0
df_large_corr = df_corr[(df_corr >= 0.7) | (df_corr <= -0.7)]
# df_large_corr = df_corr
df_large_corr = df_large_corr.dropna(how='all', axis=0)
df_large_corr = df_large_corr.dropna(how='all', axis=1)
df_large_corr = df_large_corr.fillna(0)
np.fill_diagonal(df_large_corr.values, 1)
# mask = np.zeros_like(df_large_corr)
# mask[np.triu_indices_from(mask)] = True
# fig, ax = plt.subplots()
# sns.heatmap(df_large_corr, ax=ax, annot=True, fmt='.1f', center=0,
#             vmin=-1, vmax=1, mask=mask, linecolor='k', linewidths=0.2,
#             cmap=palette)  # put
cm = sns.clustermap(data=df_large_corr, linewidths=0, cmap='coolwarm', vmax=1,
                    vmin=-1, figsize=(20, 20), annot=df_large_corr, fmt='.1f',
                    dendrogram_ratio=0.01, cbar_pos=None)
cm.ax_col_dendrogram.set_visible(False)
cm.ax_row_dendrogram.set_visible(False)

# in so for
# summary
plt.tight_layout()
plt.savefig(drop_path + "feature_correlation_heatmap_rem.png")
plt.show()

'''
Cluster map
'''
df_labels = pd.DataFrame(data_loader.label_transform(df.iloc[:, 0]))
df = df.reset_index(drop=True)
df_labels.columns = ['labels']
df_corr = pd.concat([df_labels, df.iloc[:, 5:]], axis=1).corr()
plt.figure(figsize=(20, 15))
g = sns.heatmap(df_corr, annot=True, fmt='.1f')
plt.show()



'''
Remove correlated features
'''

# remove correlated features
def correlation(dataset, threshold):
    col_corr = set()  # Set of all the names of deleted columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if (corr_matrix.iloc[i, j] >= threshold) and (corr_matrix.columns[j] not in col_corr):
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
                if colname in dataset.columns:
                    del dataset[colname] # deleting the column from the dataset
    return dataset


df_removed_features = correlation(df.iloc[:, 5:], 0.95)
listofnewfeatures = df_removed_features.columns.to_list()


'''
Infer values
'''
x_flares_idx = df_test.index[df_test['NOAA'].isin(m5_flared_NOAA)].tolist()
test_sample_x = test_loader.dataset.data[x_flares_idx].to(device)
test_sample_x = test_loader.dataset.data[41957:42147].to(device)
inferred = model(test_sample_x)
_, predicted = torch.max(inferred.data, 1)
df_inf = pd.DataFrame(inferred.cpu().detach().numpy()[:, 1], columns=list('1'))
df_pred = pd.DataFrame(predicted.cpu().detach().numpy(), columns=list('p'))
concattable = pd.concat(
    [df_pred.reset_index(drop=True), df_inf.reset_index(drop=True),
     df_test.iloc[41957:42147].reset_index(drop=True)], sort=False,
    axis=1)
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
