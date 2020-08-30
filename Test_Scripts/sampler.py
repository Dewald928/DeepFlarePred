from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import imblearn
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import ClusterCentroids
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelBinarizer
from sklearn.datasets import make_classification

sharps = ['USFLUX', 'MEANGBT', 'MEANJZH', 'MEANPOT', 'SHRGT45', 'TOTUSJH',
          'MEANGBH', 'MEANALP', 'MEANGAM', 'MEANGBZ', 'MEANJZD', 'TOTUSJZ',
          'SAVNCPP', 'TOTPOT', 'MEANSHR', 'AREA_ACR', 'R_VALUE',
          'ABSNJZH']  # 18
lorentz = ['TOTBSQ', 'TOTFX', 'TOTFY', 'TOTFZ', 'EPSX', 'EPSY', 'EPSZ']  # 7
history_features = ['Bdec', 'Cdec', 'Mdec', 'Xdec', 'Edec', 'logEdec', 'Bhis',
                    'Chis', 'Mhis', 'Xhis', 'Bhis1d', 'Chis1d', 'Mhis1d',
                    'Xhis1d', 'Xmax1d']  # 15

filepath = '../Data/Liu/z_train/'
drop_path = os.path.expanduser(
    '~/Dropbox/_Meesters/figures/sampling/')
df_train = pd.read_csv(filepath + 'normalized_training.csv')
df_val = pd.read_csv(filepath + 'normalized_validation.csv')
df_test = pd.read_csv(filepath + 'normalized_testing.csv')


def plot_2d_space(X, y, label='Classes'):
    zorders = [0,2,1]
    i = 0
    colors = ['#1F77B4', '#FF7F0E', '#00ff00']
    markers = ['o', 's', 's']
    for l, c, m in zip(np.unique(y), colors, markers):
        plt.scatter(X[y == l, 0], X[y == l, 1], c=c, label=l, marker=m,
                    zorder=zorders[i])
        i+=1
    plt.title(label)
    plt.legend(loc='upper right')
    plt.tight_layout()
    # plt.show()


# Unbalanced visual
df_train['label'].value_counts().plot(kind='bar', title='Count (target)')
plt.show()

# PCA for visualization
X = df_train.iloc[:, 5:]
y = df_train['label']

pca = PCA(n_components=2)
X = pca.fit_transform(X)

plot_2d_space(X, y, 'Imbalanced dataset (2 PCA components)')

# Random under sampling
# rus = RandomUnderSampler()
# X_rus, y_rus, id_rus = rus.fit_sample(X, y)
#
# plot_2d_space(X_rus, y_rus, 'Random under-sampling')

# cluster centroids under sampling
# cc = ClusterCentroids()
# X_cc, y_cc = cc.fit_sample(X, y)
#
# plot_2d_space(X_cc, y_cc, 'Cluster Centroids under-sampling')

# Random over sampling
ros = RandomOverSampler()
X_ros, y_ros = ros.fit_sample(X, y)

print(X_ros.shape[0] - X.shape[0], 'new random picked points')

plot_2d_space(X_ros, y_ros, 'Random over-sampling')
plt.show()
# fig, ax = plt.subplots(1,1)
# plot_resampling(X, y, ros, ax)

# SMOTE over-sampling
smote = SMOTE()
X_sm, y_sm = smote.fit_sample(X, y)

plot_2d_space(X_sm, y_sm, 'SMOTE over-sampling')
plt.show()
# over and under sampling
smt = SMOTETomek()
X_smt, y_smt = smt.fit_sample(X, y)

plot_2d_space(X_smt, y_smt, 'SMOTE + Tomek links')
plt.show()
# inverse pca?
# X_inv = pca.inverse_transform(X_cc)
#
# # Make new dataset
# df_X = pd.DataFrame(data=X_inv, columns=df_train.iloc[:, 5:].columns)
# onehot = LabelBinarizer()
# y = onehot.fit_transform(y_cc)
# df_y = pd.DataFrame(y, columns=['label'])
# df_train_sampled = pd.concat([df_y, df_X], axis=1)

'''Only use samples of ARs that flare'''
m5_flares = df_train[df_train['label'].str.match('Positive')]
m5_flared_NOAA = m5_flares['NOAA'].unique()
m5_flares_data = df_train[df_train['NOAA'].isin(m5_flared_NOAA)]
not_m5_data = df_train[~df_train['NOAA'].isin(m5_flared_NOAA)]
not_m5_data.loc[:, 'label'] = '<M5'
m5_flares_data.loc[m5_flares_data['label'] == 'Negative','label'] = '>M5, ' \
                                                                    '>24h ' \
                                                                    'before'
m5_flares_data.loc[m5_flares_data['label'] == 'Positive','label'] = '>M5, ' \
                                                                    '0-24h ' \
                                                                    'before'

df = pd.concat([not_m5_data, m5_flares_data])

# todo doen pca van die ook
pca = PCA(n_components=2)
X = pca.fit_transform(df.iloc[:, 5:])
y = df['label']

fig = plt.figure(figsize=(16,10))
plot_2d_space(X, y, 'PCA of ARs')
plt.savefig(drop_path + 'M5_PCA.png')
plt.show()

df_train_sampled = m5_flares_data

new_path = '../Data/Liu/M5_only/'
if not os.path.exists(new_path):
    os.makedirs(new_path)
df_train_sampled.to_csv(new_path + 'normalized_training.csv', index=False)
df_val.to_csv(new_path + 'normalized_validation.csv', index=False)
df_test.to_csv(new_path + 'normalized_testing.csv', index=False)
