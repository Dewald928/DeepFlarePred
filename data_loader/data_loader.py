import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np


def load_data(datafile, flare_label, series_len, start_feature, n_features,
              mask_value):
    df = pd.read_csv(datafile)
    df_values = df.values
    feature_names = list(df.columns)
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
            for k in range(5, 12):
                if float(row[k]) == 0.0:
                    has_zero_record = True
                    break
            for k in range(19, 21):
                if float(row[k]) == 0.0:
                    has_zero_record = True
                    break
            for k in range(22, 31):
                if float(row[k]) == 0.0:
                    has_zero_record = True
                    break
            for k in range(32, 37):
                if float(row[k]) == 0.0:
                    has_zero_record = True
                    break
            for k in range(40, 42):
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
                    for k in range(5, 12):
                        if float(row[k]) == 0.0:
                            has_zero_record_tmp = True
                            break
                    for k in range(19, 21):
                        if float(row[k]) == 0.0:
                            has_zero_record_tmp = True
                            break
                    for k in range(22, 31):
                        if float(row[k]) == 0.0:
                            has_zero_record_tmp = True
                            break
                    for k in range(32, 37):
                        if float(row[k]) == 0.0:
                            has_zero_record_tmp = True
                            break
                    for k in range(40, 42):
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
    # converteddata = np.eye(nclass, dtype='uint8')[encoded_Y]
    return encoded_Y