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
standardscaler.fit(df_train.iloc[:, 5:])

train_norm = pd.DataFrame(
    standardscaler.transform(df_train.iloc[:, 5:]),
    index=df_train.iloc[:, 5:].index,
    columns=df_train.iloc[:, 5:].columns)
val_norm = pd.DataFrame(
    standardscaler.transform(df_val.iloc[:, 5:]),
    index=df_val.iloc[:, 5:].index,
    columns=df_val.iloc[:, 5:].columns)
test_norm = pd.DataFrame(
    standardscaler.transform(df_test.iloc[:, 5:]),
    index=df_test.iloc[:, 5:].index,
    columns=df_test.iloc[:, 5:].columns)


# save as new datasets
# make new normalized dataset
normalized_train = pd.concat(
    [df_train.iloc[:, :5], train_norm], axis=1)
normalized_val = pd.concat([df_val.iloc[:, :5], val_norm],
                           axis=1)
normalized_test = pd.concat(
    [df_test.iloc[:, :5], test_norm], axis=1)

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
joblib.dump(standardscaler, filepath_new + '/scaler.pkl')
