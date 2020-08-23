from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

sharps = ['USFLUX', 'MEANGBT', 'MEANJZH', 'MEANPOT', 'SHRGT45', 'TOTUSJH',
          'MEANGBH', 'MEANALP', 'MEANGAM', 'MEANGBZ', 'MEANJZD', 'TOTUSJZ',
          'SAVNCPP', 'TOTPOT', 'MEANSHR', 'AREA_ACR', 'R_VALUE',
          'ABSNJZH']  # 18
lorentz = ['TOTBSQ', 'TOTFX', 'TOTFY', 'TOTFZ', 'EPSX', 'EPSY', 'EPSZ']  # 7
history_features = ['Bdec', 'Cdec', 'Mdec', 'Xdec', 'Edec', 'logEdec', 'Bhis',
                    'Chis', 'Mhis', 'Xhis', 'Bhis1d', 'Chis1d', 'Mhis1d',
                    'Xhis1d', 'Xmax1d']  # 15

filepath = '../Data/Liu_train/'
df_train = pd.read_csv(filepath + 'normalized_training.csv')
df_val = pd.read_csv(filepath + 'normalized_validation.csv')
df_test = pd.read_csv(filepath + 'normalized_testing.csv')


# Class count
count_class_0, count_class_1 = df_train['label'].value_counts()

# Divide by class
df_class_0 = df_train[df_train['label'] == 'Negative']
df_class_1 = df_train[df_train['label'] == 'Positive']

df_class_0_under = df_class_0.sample(count_class_1)
df_test_under = pd.concat([df_class_0_under, df_class_1], axis=0)

print('Random under-sampling:')
print(df_test_under['label'].value_counts())

df_test_under['label'].value_counts().plot(kind='bar', title='Count (target)')
plt.show()

new_path = '../Data/Sampled/'
if not os.path.exists(new_path):
    os.makedirs(new_path)
df_test_under.to_csv(new_path + 'normalized_training.csv', index=False)
df_val.to_csv(new_path + 'normalized_validation.csv', index=False)
df_test.to_csv(new_path + 'normalized_testing.csv', index=False)