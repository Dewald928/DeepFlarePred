from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import joblib
import os

filepath = '../Data/Liu/' + 'M5' + '/'
df_train = pd.read_csv(filepath + 'training.csv')
df_val = pd.read_csv(filepath + 'validation.csv')
df_test = pd.read_csv(filepath + 'testing.csv')

# physical features
standardscaler = StandardScaler()
standardscaler.fit(
    pd.concat([df_train], ignore_index=True))

phys_train_norm = pd.DataFrame(
    standardscaler.transform(df_train),
    index=df_train.index,
    columns=df_train.columns)
phys_val_norm = pd.DataFrame(
    standardscaler.transform(df_val),
    index=df_val.index,
    columns=df_val.columns)
phys_test_norm = pd.DataFrame(
    standardscaler.transform(df_test),
    index=df_test.index,
    columns=df_test.columns)


# save as new datasets
# make new normalized dataset
normalized_train = pd.concat(
    [df_train.iloc[:, :5], phys_train_norm], axis=1)
normalized_val = pd.concat([df_val.iloc[:, :5], phys_val_norm],
                           axis=1)
normalized_test = pd.concat(
    [df_test.iloc[:, :5], phys_test_norm], axis=1)

cols = df_train.columns.values
normalized_train = normalized_train[cols]
normalized_val = normalized_val[cols]
normalized_test = normalized_test[cols]

filepath_new = '../Data/Liu/z_train/'
if not os.path.exists(filepath_new):
    os.makedirs(filepath_new)
normalized_train.to_csv(filepath_new + 'normalized_training.csv', index=False)
normalized_val.to_csv(filepath_new + 'normalized_validation.csv', index=False)
normalized_test.to_csv(filepath_new + 'normalized_testing.csv', index=False)

# save scaler for inference later
joblib.dump(standardscaler, filepath_new + '/physcial_scaler.pkl')
joblib.dump(minmaxscaler, filepath_new + '/history_scaler.pkl')
