# Calculate metric avg and se from model

import wandb
import torch
from tabulate import tabulate
import pandas as pd
api = wandb.Api()

# load data
# load arch/model

# select 3 seed for model
# mlp oclr
# mp1 = "dewald123/liu_pytorch_MLP/a01s2pba"
# mp2 = "dewald123/liu_pytorch_MLP/7m1ej35n"
# mp3 = "dewald123/liu_pytorch_MLP/2jxi0lk6"
# MLP OCLR RELABELLED
# mp1 = "dewald123/liu_pytorch_MLP/z9oq59pc"
# mp2 = "dewald123/liu_pytorch_MLP/vlq1apay"
# mp3 = "dewald123/liu_pytorch_MLP/t9xb3vp8"
# mlp 2_500
# mp1 = "dewald123/liu_pytorch_MLP/662bk65k"
# mp2 = "dewald123/liu_pytorch_MLP/n9nofrxj"
# mp3 = "dewald123/liu_pytorch_MLP/z6u7232a"
# mlp 2_500 relabelled
mp1 = "dewald123/liu_pytorch_MLP/osl2u52s"
mp2 = "dewald123/liu_pytorch_MLP/6mmd4m7x"
mp3 = "dewald123/liu_pytorch_MLP/mukcxkol"
# MLP 1_100
# mp1 = "dewald123/liu_pytorch_MLP/16z66cnv"
# mp2 = "dewald123/liu_pytorch_MLP/bugqi272"
# mp3 = "dewald123/liu_pytorch_MLP/nzgx1hv5"
# MLP 1_100 relabelled
# mp1 = "dewald123/liu_pytorch_MLP/05gcxxjl"
# mp2 = "dewald123/liu_pytorch_MLP/ohw3rt3f"
# mp3 = "dewald123/liu_pytorch_MLP/ynlh3izo"
# CNN 1_40_7
# mp1 = "dewald123/liu_pytorch_cnn/mvzfu01i"
# mp2 = "dewald123/liu_pytorch_cnn/wub5kw1h"
# mp3 = "dewald123/liu_pytorch_cnn/87hiw98o"

th = 0.5
model_paths = [mp1, mp2, mp3]

# init metric df
coloums = ['Recall', 'Precision', 'BACC', 'HSS', 'TSS']
metrics_df = pd.DataFrame()
Train_df = pd.DataFrame(columns=coloums)
Valid_df = pd.DataFrame(columns=coloums)
Test_df = pd.DataFrame(columns=coloums)

data_dict = {'Train': Train_df, 'Valid': Valid_df,
              'Test': Test_df}
input_data = {'Train': X_train_data_tensor, 'Valid': X_valid_data_tensor,
              'Test': X_test_data_tensor}
label_data = {'Train': y_train_tr_tensor, 'Valid': y_valid_tr_tensor,
              'Test': y_test_tr_tensor}
partitions = ['Train', 'Valid', 'Test']


for i in range(len(model_paths)):
    device = 'cpu'
    model = model.to(device)
    # load model weight
    run = api.run(model_paths[i])
    weights_file = run.file('model.pt').download(replace=True)
    model.load_state_dict(torch.load(weights_file.name))

    # calculate metrics for train, val, test
    for part in partitions:
        yprob = metric.get_proba(model(input_data[part].to(device)))[:, 1]
        cm = sklearn.metrics.confusion_matrix(label_data[part],
                                              metric.to_labels(yprob, th))
        recall, precision, accuracy, bacc, tss, hss = \
            metric.calculate_metrics(cm, 2)[:6]
        series = pd.Series([recall, precision, bacc, hss, tss], index=coloums)
        # print(series)
        # add to df
        data_dict[part] = data_dict[part].append(series,
                                                 ignore_index=True)

# # calculate avg and se
for part in partitions:
    avg = data_dict[part].mean(axis=0)
    se = data_dict[part].sem(axis=0)
    std = data_dict[part].std()

    metrics_df[f'{part}_avg'] = avg
    metrics_df[f'{part}_se'] = se

# tabulate
# print(tabulate(metrics_df, tablefmt='github',
#                headers='keys', floatfmt=".4f"))

print('| Metric | Training | Validation | Test |')
print('|---|---|---|---|')
for i in range(metrics_df.shape[0]):
    print(f"| {metrics_df.index[i]} | {metrics_df.iloc[i,0]:.4f} ±"
          f" {metrics_df.iloc[i,1]:.4f} | {metrics_df.iloc[i,2]:.4f} ±"
          f" {metrics_df.iloc[i,3]:.4f} | {metrics_df.iloc[i,4]:.4f} ±"
          f" {metrics_df.iloc[i,5]:.4f} |")






