import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
import sys
import numpy as np
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from skorch import NeuralNetClassifier
from sklearn.model_selection import cross_val_predict

from collections import OrderedDict
from data_loader import CustomDataset
import wandb


"""
Dataloader
"""
def load_data(datafile, flare_label, series_len, start_feature, n_features, mask_value):
    df = pd.read_csv(datafile)
    df_values = df.values
    X = []
    y = []
    tmp = []
    X = df_values[:, start_feature:start_feature+n_features]
    y = df_values[:,0]
    X_arr = np.array(X).astype(np.float32)
    y_arr = np.array(y)
    print(X_arr.shape)
    return X_arr, y_arr


def label_transform(data):
    encoder = LabelEncoder()
    encoder.fit(data)
    encoded_Y = encoder.transform(data)
    # converteddata = np.eye(nclass, dtype='uint8')[encoded_Y]
    return encoded_Y.astype(np.long)

def preprocess_customdataset(x_val, y_val):
    # change format to tensors and create data set
    x_tensor = torch.tensor(x_val).type(torch.FloatTensor)
    y_tensor = torch.tensor(y_val).type(torch.LongTensor)
    # y_tensor = torch.nn.functional.one_hot(torch.LongTensor(y_val), 2)

    datasets = {}
    datasets = CustomDataset.CustomDataset(x_tensor, y_tensor)

    return datasets

def calculate_metrics(confusion_matrix):
    # determine skill scores
    print('Calculating skill scores: ')
    confusion_matrix = confusion_matrix.numpy()
    N = np.sum(confusion_matrix)

    recall = np.zeros(nclass)
    precision = np.zeros(nclass)
    accuracy = np.zeros(nclass)
    bacc = np.zeros(nclass)
    tss = np.zeros(nclass)
    hss = np.zeros(nclass)
    tp = np.zeros(nclass)
    fn = np.zeros(nclass)
    fp = np.zeros(nclass)
    tn = np.zeros(nclass)
    for p in range(nclass):
        tp[p] = confusion_matrix[p][p]
        for q in range(nclass):
            if q != p:
                fn[p] += confusion_matrix[p][q]
                fp[p] += confusion_matrix[q][p]
        tn[p] = N - tp[p] - fn[p] - fp[p]

        recall[p] = round(float(tp[p]) / float(tp[p] + fn[p] + 1e-6), 3)
        precision[p] = round(float(tp[p]) / float(tp[p] + fp[p] + 1e-6), 3)
        accuracy[p] = round(float(tp[p] + tn[p]) / float(N), 3)
        bacc[p] = round(
            0.5 * (float(tp[p]) / float(tp[p] + fn[p]) + float(tn[p]) / float(tn[p] + fp[p])), 3)
        hss[p] = round(2 * float(tp[p] * tn[p] - fp[p] * fn[p])
                       / float((tp[p] + fn[p]) * (fn[p] + tn[p])
                               + (tp[p] + fp[p]) * (fp[p] + tn[p])), 3)
        tss[p] = round(
            (float(tp[p]) / float(tp[p] + fn[p] + 1e-6) - float(fp[p]) / float(fp[p] + tn[p] + 1e-6)), 3)

    print("tss: " + str(tss))
    print("hss: " + str(hss))
    print("bacc: " + str(bacc))
    print("accuracy: " + str(accuracy))
    print("precision: " + str(precision))
    print("recall: " + str(recall))

    return recall, precision, accuracy, bacc, tss, hss, tp, fn, fp, tn


"""
MLP model
"""
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layer_dim, dropout=0.5,):
        super(MLP, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        layers = []
        layers += [nn.Linear(input_dim, hidden_dim)]

        for i in range(layer_dim):
            layers += [nn.Linear(hidden_dim, hidden_dim)]
            layers += [nn.ReLU()]

        layers += [nn.Linear(hidden_dim, output_dim)]

        self.network = nn.Sequential(*layers)


    def forward(self, X, **kwargs):
        X = F.softmax(self.network(X), dim=-1)
        return X

def train(model, device, train_loader, optimizer, epoch, criterion):
    model.train()
    confusion_matrix = torch.zeros(nclass, nclass)
    loss_epoch = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss_epoch += criterion(output, target).item()
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(output.data, 1)

        for t, p in zip(target.view(-1), predicted.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

    loss_epoch /= len(train_loader.dataset)
    print("Training Scores:")
    recall, precision, accuracy, bacc, tss, hss, tp, fn, fp, tn = calculate_metrics(confusion_matrix)

    wandb.log({
        "Training_Accuracy": accuracy[0],
        "Training_TSS": tss[0],
        "Training_HSS": hss[0],
        "Training_BACC": bacc[0],
        "Training_Precision": precision[0],
        "Training_Recall": recall[0],
        "Training_Loss": loss_epoch}, step=epoch)


def validate(model, device, valid_loader, criterion, epoch):
    model.eval()
    valid_loss = 0
    correct = 0
    confusion_matrix = torch.zeros(nclass, nclass)

    example_images = []
    with torch.no_grad():
        for data, target in valid_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            valid_loss += criterion(output, target).item()
            # get the index of the max log-probability
            _, predicted = torch.max(output.data, 1)
            correct += predicted.eq(target.view_as(predicted)).sum().item()

            for t, p in zip(target.view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

    valid_loss /= len(valid_loader.dataset)
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        valid_loss, correct, len(valid_loader.dataset),
        100. * correct / len(valid_loader.dataset)))

    # # validation conf matrix
    # print(confusion_matrix)
    # # per class accuracy
    # print(confusion_matrix.diag() / confusion_matrix.sum(1))

    print("Validation Scores:")
    recall, precision, accuracy, bacc, tss, hss, tp, fn, fp, tn = calculate_metrics(confusion_matrix)

    wandb.log({
        "Validation_Accuracy": accuracy[0],
        "Validation_TSS": tss[0],
        "Validation_HSS": hss[0],
        "Validation_BACC": bacc[0],
        "Validation_Precision": precision[0],
        "Validation_Recall": recall[0],
        "Validation_Loss": valid_loss}, step=epoch)



if __name__ == '__main__':
    wandb.init(project='Liu_MLP')

    # parse hyperparameters
    parser = argparse.ArgumentParser(description='Deep Flare Prediction')
    parser.add_argument('--batch_size', type=int, default=512, metavar='N',
                        help='input batch size for training (default: 256)')
    parser.add_argument('--test_batch_size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=40, metavar='N',
                        help='number of epochs to train (default: 15)')
    parser.add_argument('--learning_rate', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log_interval', type=int, default=30, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--flare_label', default="M",
                        help='Types of flare class (default: M-Class')
    parser.add_argument('--layer_dim', type=int, default=5, metavar='N',
                        help='how many hidden layers (default: 5)')
    parser.add_argument('--hidden_dim', type=int, default=1000, metavar='N',
                        help='how many nodes in layers (default: 64)')
    parser.add_argument('--dropout', type=float, default=0.4, metavar='M',
                        help='percentage dropout (default: 0.4)')
    parser.add_argument('--weight_decay', type=float, default=0.0001, metavar='LR',
                        help='L2 regularizing (default: 0.0001)')
    parser.add_argument('--rnn_module', default="GRU",
                        help='Types of rnn (default: LSTM')
    args = parser.parse_args()
    wandb.config.update(args)

    # initialize parameters
    filepath = './Data/Liu/' + args.flare_label + '/'
    num_of_fold = 10
    n_features = 0
    if args.flare_label == 'M5':
        n_features = 20
    elif args.flare_label == 'M':
        n_features = 22
    elif args.flare_label == 'C':
        n_features = 14

    # initialize parameters
    start_feature = 5
    mask_value = 0
    nclass = 2

    # GPU check
    use_cuda = args.cuda and torch.cuda.is_available()
    if args.cuda == True and torch.cuda.is_available():
        print("Cuda enabled and available")
    elif args.cuda == True and torch.cuda.is_available() == False:
        print("Cuda enabled not not available, CPU used.")
    elif args.cuda == False:
        print("Cuda disabled")

    # set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # And this
    np.random.seed(args.seed)

    # setup dataloaders
    X_train_data, y_train_data = load_data(datafile=filepath + 'normalized_training.csv',
                                           flare_label=args.flare_label, series_len=args.layer_dim,
                                           start_feature=start_feature, n_features=n_features,
                                           mask_value=mask_value)

    X_valid_data, y_valid_data = load_data(datafile=filepath + 'normalized_validation.csv',
                                           flare_label=args.flare_label, series_len=args.layer_dim,
                                           start_feature=start_feature, n_features=n_features,
                                           mask_value=mask_value)

    X_test_data, y_test_data = load_data(datafile=filepath + 'normalized_testing.csv',
                                         flare_label=args.flare_label, series_len=args.layer_dim,
                                         start_feature=start_feature, n_features=n_features,
                                         mask_value=mask_value)

    y_train_tr = label_transform(y_train_data)
    y_valid_tr = label_transform(y_valid_data)
    y_test_tr = label_transform(y_test_data)

    # ready custom dataset
    datasets = {}
    datasets['train'] = preprocess_customdataset(X_train_data, y_train_tr)
    datasets['valid'] = preprocess_customdataset(X_valid_data, y_valid_tr)
    datasets['test'] = preprocess_customdataset(X_test_data, y_test_tr)

    train_loader = torch.utils.data.DataLoader(datasets['train'], args.batch_size,
                                               shuffle=False, drop_last=False)
    valid_loader = torch.utils.data.DataLoader(datasets['valid'], args.batch_size,
                                               shuffle=False, drop_last=False)
    test_loader = torch.utils.data.DataLoader(datasets['test'], args.batch_size,
                                              shuffle=False, drop_last=False)

    #make model
    device = torch.device("cuda" if use_cuda else "cpu")
    model = MLP(input_dim=n_features, hidden_dim=args.hidden_dim, output_dim=nclass, layer_dim=args.layer_dim).to(device)
    wandb.watch(model, log='all')

    # optimizers
    class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train_data), y_train_data)
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(device))  # weighted cross entropy
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # print model parameters
    print(len(list(model.parameters())))
    for i in range(len(list(model.parameters()))):
        print(list(model.parameters())[i].size())

    print("Training Network...")
    for epoch in range(args.epochs):
        train(model, device, train_loader, optimizer, epoch, criterion)
        validate(model, device, valid_loader, criterion, epoch)

    '''
    K-fold cross validation
    '''
    net = NeuralNetClassifier(
        model,
        max_epochs=args.epochs,
        lr=args.learning_rate,
        device=device,
        train_split=None
    )

    from sklearn.model_selection import cross_val_predict

    y_pred = cross_val_predict(net, X_train_data, y_train_tr, cv=5)

    # net.fit(X_train_data, y_train_tr)

    print("finished")