import sklearn
import wandb
import numpy as np

from skorch.callbacks import *


def get_metric(y_true, y_pred, nclass, metric_name='tss'):
    # print('Calculating skill scores: ')
    confusion_matrix = []
    confusion_matrix = sklearn.metrics.confusion_matrix(y_true, y_pred)
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

        recall[p] = round(float(tp[p]) / float(tp[p] + fn[p] + 1e-6), 4)
        precision[p] = round(float(tp[p]) / float(tp[p] + fp[p] + 1e-6), 4)
        accuracy[p] = round(float(tp[p] + tn[p]) / float(N), 3)
        bacc[p] = round(0.5 * (
                float(tp[p]) / float(tp[p] + fn[p]) + float(tn[p]) / float(
            tn[p] + fp[p]) + 1e-6), 4)
        hss[p] = round(2 * float(tp[p] * tn[p] - fp[p] * fn[p]) / float(
            (tp[p] + fn[p]) * (fn[p] + tn[p]) + (tp[p] + fp[p]) * (
                    fp[p] + tn[p])), 4)
        tss[p] = round((float(tp[p]) / float(tp[p] + fn[p] + 1e-6) - float(
            fp[p]) / float(fp[p] + tn[p] + 1e-6)), 4)

    if metric_name == 'recall':
        return recall[0]
    if metric_name == 'precision':
        return precision[0]
    if metric_name == 'accuracy':
        return accuracy[0]
    if metric_name == 'bacc':
        return bacc[0]
    if metric_name == 'tss':
        return tss[0]
    if metric_name == 'hss':
        return hss[0]
    else:
        return recall, precision, accuracy, bacc, tss, hss, tp, fn, fp, tn


def get_tss(y_true, y_pred):
    tss = get_metric(y_true, y_pred, 2, metric_name='tss')
    return tss


def get_hss(y_true, y_pred):
    return get_metric(y_true, y_pred, 2, metric_name='hss')


class LoggingCallback(Callback):
    def __init__(self, ):
        super(LoggingCallback, self).__init__()

    # def initialize(self):
    #     wandb.log(step=0)

    def on_train_begin(self, net, X=None, y=None, **kwargs):
        wandb.log(step=0)

    def on_train_end(self, net, X=None, y=None, **kwargs):
        h = net.history
        best_tss_epoch = np.argmax(h[:, 'valid_tss'])
        best_tss = np.max(h[:, 'valid_tss'])
        wandb.log({'Best_Validation_TSS': best_tss,
                   'Best_Validation_epoch': best_tss_epoch})

    def on_epoch_end(self, net, dataset_trn=None, dataset_vld=None, **kwargs):
        h = net.history[-1]
        wandb.log(
            {'Train_Loss': h['train_loss'], 'Validation_TSS': h['valid_tss'],
             'Validation_HSS': h['valid_hss'],
             'Validation_Loss': h['valid_loss']}, step=h['epoch'])


class LoadBestCP(Callback):
    def __init__(self, checkpoint):
        self.checkpoint = checkpoint
        super(LoadBestCP, self).__init__()

    def on_train_end(self, net, X=None, y=None, **kwargs):
        net.load_params(checkpoint=self.checkpoint)


# counter = 0
#
#
# class MyNet():
#     def fit(self, X, y, **fit_params):
#         global counter
#         counter += 1  # increase the counter with each fit call
#         return super().fit(X, y, **fit_params)


class MyCheckpoint(Checkpoint):
    def get_formatted_files(self, *args, **kwargs):
        files_dict = super().get_formatted_files(*args, **kwargs)
        # add code to modify the values of the dict using "counter"
        return files_dict