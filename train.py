from __future__ import print_function
import argparse

from data_loader import data_loaders
import model.model as model_arch
import model.loss as model_loss
import model.metrics as model_metric

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import pandas as pd

from sklearn.utils import class_weight
import sys
import numpy as np

import wandb


def main():
    # Initialize Weight and Biases Project
    wandb.init(project='DeepFlarePred')

    # Setup Settings
    n_features = 14
    start_feature = 5
    mask_value = 0
    series_len = 7
    nclass = 2
    hidden_dim = 24
    thlistsize = 201
    thlist = np.linspace(0, 1, thlistsize)

    # Training settings
    parser = argparse.ArgumentParser(description='DeepFlarePred')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--flare-label', default="M",
                        help='Types of flare class (default: M-Class')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    filepath = './Data/Liu/' + args.flare_label + '/'
    num_of_fold = 10
    n_features = 0
    if args.flare_label == 'M5':
        n_features = 20
    elif args.flare_label == 'M':
        n_features = 22
    elif args.flare_label == 'C':
        n_features = 14

    wandb.config.update(args)

    # Set seed
    torch.manual_seed(args.seed)

    # Setup Dataloader
    dataloader = data_loaders.DataLoader(filepath, args.flare_label, series_len, start_feature, n_features,
                                         mask_value, args.batch_size)

    # Build model architecture
    model = model_arch.LSTMModel(n_features, hidden_dim=hidden_dim, layer_dim=series_len, output_dim=nclass)
    wandb.watch(model, log='all')

    # Build optimizer
    class_weights = class_weight.compute_class_weight('balanced', np.unique(dataloader.y_train_data),
                                                      dataloader.y_train_data)

    # weighted cross entropy, because it's unbalanced
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(len(list(model.parameters())))

    for i in range(len(list(model.parameters()))):
        print(list(model.parameters())[i].size())





if __name__ == '__main__':
    main()