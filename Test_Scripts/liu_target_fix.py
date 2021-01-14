import pandas as pd
import numpy as np
import datetime
import os
from data_loader import data_loader
import sunpy
from sunpy.time import TimeRange
import sunpy.instr.goes
from datetime import timedelta
from datetime import datetime as dt_obj


def parse_tai_string(tstr):
    year = int(tstr[:4])
    month = int(tstr[5:7])
    day = int(tstr[8:10])
    hour = int(tstr[11:13])
    minute = int(tstr[14:16])
    return dt_obj(year, month, day, hour, minute)


drop_path = os.path.expanduser(
    '~/Dropbox/_Meesters/figures/features_inspect/')
filepath = '../Data/Liu/z_train/'

# get flare events
t_start = "2010-05-01"
t_end = "2018-05-11"
time_range = TimeRange(t_start, t_end)
listofresults = sunpy.instr.goes.get_goes_event_list(time_range, 'M5')
listofresults = pd.DataFrame(listofresults)
listofresults = listofresults[listofresults['noaa_active_region'] != 0]
pd.DataFrame(listofresults).to_csv('../Data/GOES/all_flares_list.csv')
listofresults['peak_time'] = listofresults['peak_time'].astype(str).apply(parse_tai_string)

splits = ['training', 'validation', 'testing']

for split in splits:
    df = pd.read_csv(filepath + 'normalized_{}.csv'.format(split))

    # get positive labels
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%dT%H:%M:%S.%fZ')
    m5_flares = df[df['label'].str.match('Positive')]
    m5_flared_NOAA = m5_flares['NOAA'].unique()
    m5_flares_data = df[df['NOAA'].isin(m5_flared_NOAA)].sort_values(by=['NOAA','timestamp'])

    # check if spans 24hours for that noaa
    # for each flare event
    for i in range(listofresults.shape[0]):
        peak_time = listofresults['peak_time'].iloc[i]
        noaa_ar = listofresults['noaa_active_region'].iloc[i]
        current_ar_df = m5_flares_data[m5_flares_data['NOAA'] == noaa_ar]
        time_before = peak_time - timedelta(hours=24)
        # get samples between peak and 24h before, label as positive
        mask = ((current_ar_df['timestamp'] > time_before) &
             (current_ar_df['timestamp'] < peak_time))
        time_before_flare_df = current_ar_df.loc[mask]
        df.loc[df.index[time_before_flare_df.index], 'label'] = \
            'Positive'

    # save new dataset
    filepath_new = '../Data/Liu/z_train_relabelled/'
    if not os.path.exists(filepath_new):
        os.makedirs(filepath_new)
    df.to_csv(filepath_new + f'normalized_{split}.csv', index=False)
