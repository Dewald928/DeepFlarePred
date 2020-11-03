import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from itertools import accumulate
from torch.nn.utils.rnn import pad_sequence
import torch

sharps = ['USFLUX', 'MEANGBT', 'MEANJZH', 'MEANPOT', 'SHRGT45', 'TOTUSJH',
          'MEANGBH', 'MEANALP', 'MEANGAM', 'MEANGBZ', 'MEANJZD', 'TOTUSJZ',
          'SAVNCPP', 'TOTPOT', 'MEANSHR', 'AREA_ACR', 'R_VALUE', 'ABSNJZH']
lorentz = ['TOTBSQ', 'TOTFX', 'TOTFY', 'TOTFZ', 'EPSX', 'EPSY', 'EPSZ']
history_features = ['Bdec', 'Cdec', 'Mdec', 'Xdec', 'Edec', 'logEdec', 'Bhis',
                    'Chis', 'Mhis', 'Xhis', 'Bhis1d', 'Chis1d', 'Mhis1d',
                    'Xhis1d', 'Xmax1d']
physical_features = sharps + lorentz


def load_data(datafile, flare_label, series_len, start_feature, n_features,
              mask_value, feature_list=None):
    df = pd.read_csv(datafile)
    df = df.sort_values(by=['NOAA', 'timestamp'])  # I added this, valid?
    df, n_features = feature_select(feature_list, df, start_feature, n_features)
    feature_names = df.columns
    physical_features_idx = []
    for i in range(len(physical_features)):
        try:
            physical_features_idx.append(feature_names.get_loc(
                physical_features[i]))
        except:
            continue
    df_values = df.values
    X = []
    y = []
    tmp = []
    for k in range(start_feature, start_feature + n_features):
        tmp.append(mask_value)
    for idx in range(0, len(df_values)):
        each_series_data = []
        row = df_values[idx]
        if flare_label == 'M5' and row[1][0] == 'M' and float(
                row[1][1:]) >= 5.0:
            label = 'X'
        else:
            label = row[1][0]
        if flare_label == 'M' and label == 'X':
            label = 'M'
        if flare_label == 'C' and (label == 'X' or label == 'M'):
            label = 'C'
        if flare_label == 'B' and (
                label == 'X' or label == 'M' or label == 'C'):
            label = 'B'
        if flare_label == 'M5' and (
                label == 'M' or label == 'C' or label == 'B'):
            label = 'N'
        if flare_label == 'M' and (label == 'B' or label == 'C'):
            label = 'N'
        if flare_label == 'C' and label == 'B':
            label = 'N'
        has_zero_record = False
        # if at least one of the 25 physical feature values is missing,
        # then discard it.
        if flare_label == 'C':
            if float(row[5]) == 0.0:
                has_zero_record = True
            if float(row[7]) == 0.0:
                has_zero_record = True
            for k in range(9, 13):
                if float(row[k]) == 0.0:
                    has_zero_record = True
                    break
            for k in range(14, 16):
                if float(row[k]) == 0.0:
                    has_zero_record = True
                    break
            for k in range(18, 21):
                if float(row[k]) == 0.0:
                    has_zero_record = True
                    break
            if float(row[22]) == 0.0:
                has_zero_record = True
            for k in range(24, 33):
                if float(row[k]) == 0.0:
                    has_zero_record = True
                    break
            for k in range(38, 42):
                if float(row[k]) == 0.0:
                    has_zero_record = True
                    break
        elif flare_label == 'M':
            for k in range(5, 10):
                if float(row[k]) == 0.0:
                    has_zero_record = True
                    break
            for k in range(13, 16):
                if float(row[k]) == 0.0:
                    has_zero_record = True
                    break
            if float(row[19]) == 0.0:
                has_zero_record = True
            if float(row[21]) == 0.0:
                has_zero_record = True
            for k in range(23, 30):
                if float(row[k]) == 0.0:
                    has_zero_record = True
                    break
            for k in range(31, 33):
                if float(row[k]) == 0.0:
                    has_zero_record = True
                    break
            for k in range(34, 37):
                if float(row[k]) == 0.0:
                    has_zero_record = True
                    break
            for k in range(39, 41):
                if float(row[k]) == 0.0:
                    has_zero_record = True
                    break
            if float(row[42]) == 0.0:
                has_zero_record = True
        elif flare_label == 'M5':
            for k in physical_features_idx:
                if float(row[k]) == 0.0:
                    has_zero_record = True
                    break

        if has_zero_record is False:
            cur_noaa_num = int(row[3])
            each_series_data.append(
                row[start_feature:start_feature + n_features].tolist())
            itr_idx = idx - 1
            while itr_idx >= 0 and len(each_series_data) < series_len:
                prev_row = df_values[itr_idx]
                prev_noaa_num = int(prev_row[3])
                if prev_noaa_num != cur_noaa_num:
                    break
                has_zero_record_tmp = False
                if flare_label == 'C':
                    if float(row[5]) == 0.0:
                        has_zero_record_tmp = True
                    if float(row[7]) == 0.0:
                        has_zero_record_tmp = True
                    for k in range(9, 13):
                        if float(row[k]) == 0.0:
                            has_zero_record_tmp = True
                            break
                    for k in range(14, 16):
                        if float(row[k]) == 0.0:
                            has_zero_record_tmp = True
                            break
                    for k in range(18, 21):
                        if float(row[k]) == 0.0:
                            has_zero_record_tmp = True
                            break
                    if float(row[22]) == 0.0:
                        has_zero_record_tmp = True
                    for k in range(24, 33):
                        if float(row[k]) == 0.0:
                            has_zero_record_tmp = True
                            break
                    for k in range(38, 42):
                        if float(row[k]) == 0.0:
                            has_zero_record_tmp = True
                            break
                elif flare_label == 'M':
                    for k in range(5, 10):
                        if float(row[k]) == 0.0:
                            has_zero_record_tmp = True
                            break
                    for k in range(13, 16):
                        if float(row[k]) == 0.0:
                            has_zero_record_tmp = True
                            break
                    if float(row[19]) == 0.0:
                        has_zero_record_tmp = True
                    if float(row[21]) == 0.0:
                        has_zero_record_tmp = True
                    for k in range(23, 30):
                        if float(row[k]) == 0.0:
                            has_zero_record_tmp = True
                            break
                    for k in range(31, 33):
                        if float(row[k]) == 0.0:
                            has_zero_record_tmp = True
                            break
                    for k in range(34, 37):
                        if float(row[k]) == 0.0:
                            has_zero_record_tmp = True
                            break
                    for k in range(39, 41):
                        if float(row[k]) == 0.0:
                            has_zero_record_tmp = True
                            break
                    if float(row[42]) == 0.0:
                        has_zero_record_tmp = True
                elif flare_label == 'M5':
                    for k in physical_features_idx:
                        if float(row[k]) == 0.0:
                            has_zero_record_tmp = True
                            break
                if len(
                        each_series_data) < series_len and \
                        has_zero_record_tmp is True:
                    each_series_data.insert(0, tmp)

                if len(
                        each_series_data) < series_len and \
                        has_zero_record_tmp is False:
                    each_series_data.insert(0, prev_row[
                                               start_feature:start_feature +
                                                             n_features].tolist())
                itr_idx -= 1

            while len(each_series_data) > 0 and len(
                    each_series_data) < series_len:
                each_series_data.insert(0, tmp)

            if len(each_series_data) > 0:
                X.append(np.array(each_series_data).reshape(series_len,
                                                            n_features).tolist())
                y.append(label)
    # select features
    X_arr = np.array(X)
    y_arr = np.array(y)
    print(X_arr.shape)
    return X_arr, y_arr


def get_feature_names(datafile):
    df = pd.read_csv(datafile)
    feature_names = list(df.columns)
    return feature_names


def label_transform(data):
    encoder = LabelEncoder()
    encoder.fit(data)
    encoded_Y = encoder.transform(data)
    return encoded_Y


def partition_10_folds(X, y, num_of_fold):
    num = len(X)
    index = [i for i in range(num)]
    # np.random.seed(123)
    np.random.shuffle(index)
    X_output = []
    y_output = []
    num_in_each_fold = round(num / num_of_fold)
    for i in range(num_of_fold):
        if i == (num_of_fold - 1):
            idx = index[num_in_each_fold * (num_of_fold - 1):]
        else:
            idx = index[num_in_each_fold * i: num_in_each_fold * (i + 1)]
        X_output.append(X[idx])
        y_output.append(y[idx])
    return X_output, y_output


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


def load_reg_data(datafile, flare_label, series_len, start_feature, n_features,
              mask_value):
    df = pd.read_csv(datafile)
    df = df.sort_values(by=['NOAA', 'timestamp'])  # I added this, valid?
    df_values = df.values

    flux = df['flare'].apply(flare_to_flux)
    flux.name = 'flux'
    df_flux = pd.concat([flux, df], axis=1)
    df_flux = df_flux.drop(columns='label')

    X = df_flux.iloc[:, start_feature: start_feature+n_features]
    X = create_inout_sequences(X, series_len, n_features)
    y = df_flux['flux']

    X_arr = np.array(X)
    y_arr = np.array(y)
    print(X_arr.shape)
    return X_arr, y_arr


def create_inout_sequences(input_data, tw, n_features):
    inout_seq = []
    L = len(input_data)
    for i in range(1, L+1):
        train_seq = []
        if i <= tw:
            train_seq = np.zeros((tw-i, n_features))
            train_seq = np.concatenate((train_seq, input_data[0:i]), axis=0)
        elif i > tw:
            train_seq = input_data[i-tw:i]
        inout_seq.append(np.asarray(train_seq))
    return inout_seq


def feature_select(feature_list, df, start_feature, n_features):
    # pass the required feature names, returns sorted df
    if feature_list == None:
        return df, n_features
    else:
        df_config = df.iloc[:, :start_feature]
        df_new_features = df.loc[:, feature_list]
        df_new = pd.concat([df_config, df_new_features], axis=1)
        n_features = df_new.shape[1] - start_feature
        # global n_features
        # n_features = df_new_features.shape[1]
        return df_new, n_features



