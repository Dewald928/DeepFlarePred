import sklearn
import wandb
import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from skorch.callbacks import *
from model import metric


def get_metric(y_true, y_pred, nclass, metric_name='tss'):
    # print('Calculating skill scores: ')
    confusion_matrix = []

    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    pr_auc = auc(recall, precision)


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
    if metric_name == 'pr_auc':
        return pr_auc
    else:
        return recall, precision, accuracy, bacc, tss, hss, tp, fn, fp, tn


def get_tss(y_true, y_pred):
    tss = get_metric(y_true, y_pred, 2, metric_name='tss')
    return tss


def get_hss(y_true, y_pred):
    return get_metric(y_true, y_pred, 2, metric_name='hss')

def get_pr_auc(y_true, y_pred):
    return get_metric(y_true, y_pred, 2, metric_name='pr_auc')


class LoggingCallback(Callback):
    def __init__(self, test_inputs, test_labels):
        self.test_inputs = test_inputs
        self.test_labels = test_labels
        super(LoggingCallback, self).__init__()

    # def initialize(self):
    #     wandb.log(step=0)

    # def on_train_begin(self, net, X=None, y=None, **kwargs):
    #     wandb.log(step=0)

    def on_train_end(self, net, X=None, y=None, **kwargs):
        h = net.history
        best_tss_epoch = np.argmax(h[:, 'valid_tss'])
        best_val_tss = np.max(h[:, 'valid_tss'])
        best_train_tss = h[best_tss_epoch, 'train_tss']
        wandb.log({'Best_Validation_TSS': best_val_tss,
                   'Best_Train_TSS': best_train_tss,
                   'Best_Validation_epoch': best_tss_epoch})

    def on_epoch_end(self, net, dataset_trn=None, dataset_vld=None, **kwargs):
        y_test = net.predict(self.test_inputs)
        tss_test_score = get_tss(self.test_labels, y_test)
        h = net.history[-1]
        wandb.log({'Training_Loss': h['train_loss'],
                   'Validation_TSS': h['valid_tss'],
                   'Training_TSS': h['train_tss'],
                   'Validation_HSS': h['valid_hss'],
                   'Validation_Loss': h['valid_loss'],
                   'Validation_BSS': h['valid_bss'],
                   # 'Validation_PR_AUC': h['valid_pr_auc'],
                   'Test_TSS_curve': tss_test_score,
                   'Epoch': h['epoch']})


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


def get_bss(net, ds, y=None):
    y_prob = net.predict_proba(ds)[:,1]
    bss = metric.get_bss(y_prob,y)
    return bss