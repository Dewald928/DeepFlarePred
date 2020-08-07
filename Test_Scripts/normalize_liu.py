from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import joblib

sharps = ['USFLUX', 'MEANGBT', 'MEANJZH', 'MEANPOT', 'SHRGT45', 'TOTUSJH',
          'MEANGBH', 'MEANALP', 'MEANGAM', 'MEANGBZ', 'MEANJZD', 'TOTUSJZ',
          'SAVNCPP', 'TOTPOT', 'MEANSHR', 'AREA_ACR', 'R_VALUE',
          'ABSNJZH']  # 18
lorentz = ['TOTBSQ', 'TOTFX', 'TOTFY', 'TOTFZ', 'EPSX', 'EPSY', 'EPSZ']  # 7
history_features = ['Bdec', 'Cdec', 'Mdec', 'Xdec', 'Edec', 'logEdec', 'Bhis',
                    'Chis', 'Mhis', 'Xhis', 'Bhis1d', 'Chis1d', 'Mhis1d',
                    'Xhis1d', 'Xmax1d']  # 15

filepath = '../Data/Liu/' + 'M5' + '/'
df_train = pd.read_csv(filepath + 'training.csv')
df_val = pd.read_csv(filepath + 'validation.csv')
df_test = pd.read_csv(filepath + 'testing.csv')

# physical features
standardscaler = StandardScaler()
standardscaler.fit(
    pd.concat([df_train.loc[:, sharps + lorentz]], ignore_index=True))

phys_train_norm = pd.DataFrame(
    standardscaler.transform(df_train.loc[:, sharps + lorentz]),
    index=df_train.loc[:, sharps + lorentz].index,
    columns=df_train.loc[:, sharps + lorentz].columns)
phys_val_norm = pd.DataFrame(
    standardscaler.transform(df_val.loc[:, sharps + lorentz]),
    index=df_val.loc[:, sharps + lorentz].index,
    columns=df_val.loc[:, sharps + lorentz].columns)
phys_test_norm = pd.DataFrame(
    standardscaler.transform(df_test.loc[:, sharps + lorentz]),
    index=df_test.loc[:, sharps + lorentz].index,
    columns=df_test.loc[:, sharps + lorentz].columns)

# history features
minmaxscaler = MinMaxScaler()
minmaxscaler.fit(
    pd.concat([df_train.loc[:, history_features]], ignore_index=True))

his_train_norm = pd.DataFrame(
    minmaxscaler.transform(df_train.loc[:, history_features]),
    index=df_train.loc[:, history_features].index,
    columns=df_train.loc[:, history_features].columns)
his_val_norm = pd.DataFrame(
    minmaxscaler.transform(df_val.loc[:, history_features]),
    index=df_val.loc[:, history_features].index,
    columns=df_val.loc[:, history_features].columns)
his_test_norm = pd.DataFrame(
    minmaxscaler.transform(df_test.loc[:, history_features]),
    index=df_test.loc[:, history_features].index,
    columns=df_test.loc[:, history_features].columns)

# save as new datasets
# make new normalized dataset
normalized_train = pd.concat(
    [df_train.iloc[:, :5], phys_train_norm, his_train_norm], axis=1)
normalized_val = pd.concat([df_val.iloc[:, :5], phys_val_norm, his_val_norm],
                           axis=1)
normalized_test = pd.concat(
    [df_test.iloc[:, :5], phys_test_norm, his_test_norm], axis=1)

filepath_new = '../Data/Liu_train/'
normalized_train.to_csv(filepath_new + 'normalized_training.csv', index=False)
normalized_val.to_csv(filepath_new + 'normalized_validation.csv', index=False)
normalized_test.to_csv(filepath_new + 'normalized_testing.csv', index=False)

# save scaler for inference later
joblib.dump(standardscaler, filepath_new + '/physcial_scaler.pkl')
joblib.dump(minmaxscaler, filepath_new + '/history_scaler.pkl')
