import pandas as pd
import torch
import main_TCN_Liu

filepath = './Data/Liu/' + args.flare_label+ '/'
df = pd.read_csv(filepath + 'normalized_testing.csv')

# get x flares
m5_flares_test = df[df['flare'].str.match('M5') |
                    df['flare'].str.match('M6') |
                    df['flare'].str.match('M7') |
                    df['flare'].str.match('M8') |
                    df['flare'].str.match('M9') |
                    df['flare'].str.match('X')]
m5_flared_NOAA = m5_flares_test['NOAA'].unique()

'''
Infer values
'''
df = pd.read_csv(filepath + 'normalized_testing.csv')
x_flares_data = df[df['NOAA'].isin(m5_flared_NOAA)]
x_flares_idx =df.index[df['NOAA'].isin(m5_flared_NOAA)].tolist()
test_sample_x = test_loader.dataset.data[x_flares_idx].to(device)
inferred = model(test_sample_x)
_, predicted = torch.max(inferred.data, 1)
df_inf = pd.DataFrame(inferred.cpu().detach().numpy(), columns=list('01'))
df_pred = pd.DataFrame(predicted.cpu().detach().numpy(), columns=list('p'))
concattable = pd.concat(
    [df_pred.reset_index(drop=True), df_inf.reset_index(drop=True),
     x_flares_data.reset_index(drop=True)], sort=False, axis=1)

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
writer.close()
# tensorboard --logdir=runs
