import pandas as pd

filepath = './Data/Liu/' + 'M5.0' + '/'
df = pd.read_csv(filepath + 'normalized_testing.csv')

# get x flares
x_flares_test = df[df['flare'].str.match('X')]
df[df['NOAA'] == 12297]