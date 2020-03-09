import numpy as np
import sklearn
import torch
import wandb

# precision-recall curve and f1
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from matplotlib import pyplot as plt

from utils import confusion_matrix_plot


def calculate_metrics(confusion_matrix, nclass):
    # determine skill scores
    # confusion_matrix = confusion_matrix.numpy()
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
    mcc = np.zeros(nclass)
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
                    tn[p] + fp[p])), 4)
        hss[p] = round(2 * float(tp[p] * tn[p] - fp[p] * fn[p]) / float(
            (tp[p] + fn[p]) * (fn[p] + tn[p]) + (tp[p] + fp[p]) * (
                    fp[p] + tn[p])), 4)
        tss[p] = round((float(tp[p]) / float(tp[p] + fn[p] + 1e-6) - float(
            fp[p]) / float(fp[p] + tn[p] + 1e-6)), 4)
        mcc[p] = round(float(tp[p] * tn[p] - fp[p] * fn[p]) / float(np.sqrt(
            (tp[p] + fp[p]) * (tp[p] + fn[p]) * (tn[p] + fp[p]) * (
                    tn[p] + fn[p])) + 1e-6), 4)

    return recall[1], precision[1], accuracy[1], bacc[1], tss[1], hss[1], tp[
        1], fn[1], fp[1], tn[1], mcc[1]


def get_roc(model, yhat, ytrue, device, dataset='Test'):
    model.eval()
    fig = plt.figure()
    ytrue = ytrue.cpu().detach().numpy()
    with torch.no_grad():
        # calculate roc curve
        # retrieve just the probabilities for the positive class
        pos_probs = yhat[:, 1]
        # get predictied labels
        _, ypred = torch.max(torch.tensor(yhat), 1)
        ypred = ypred.cpu().detach().numpy()

        # calculate roc curve for model
        fpr, tpr, thresholds = roc_curve(ytrue, pos_probs)
        # calculate roc auc
        roc_auc = roc_auc_score(ytrue, pos_probs)
        print('Model ROC AUC %.3f' % roc_auc)
        # get the best threshold

        J = tpr - fpr
        ix = np.argmax(J)
        print('Best Threshold=%f, J stat=%.3f' % (thresholds[ix], J[ix]))
        # probs = yhat[:, 1]
        # # define thresholds
        # thresholds = np.arange(0, 1, 0.01)
        # # evaluate each threshold
        # scores = [f1_score(ytrue, to_labels(probs, t)) for t in
        #           thresholds]
        # # get best threshold
        # ix = np.argmax(scores)
        # print('Best Threshold=%f, F-Score=%.3f' % (thresholds[ix], scores[ix]))

        # plot model roc curve
        # plot no skill roc curve
        plt.plot([0, 1], [0, 1], linestyle='--', label='No Skill', zorder=1)
        plt.plot(fpr, tpr, marker='.', label='AUC '+str(round(roc_auc, 3)),
                 zorder=2)
        plt.scatter(fpr[ix], tpr[ix], marker='o', color='black',
                    label='Best', zorder=3)
        # axis labels
        plt.title(dataset+ ' ROC curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        # show the legend
        plt.legend()
        # show the plot
        fig.show()
        wandb.log({dataset + ' ROC curve': wandb.Image(fig),
                   dataset + ' ROC_AUC': roc_auc})

        return roc_auc


def plot_confusion_matrix(yhat, ytrue, dataset):
    ytrue = ytrue.cpu().detach().numpy()
    # retrieve just the probabilities for the positive class
    pos_probs = yhat[:, 1]
    # get predicted labels
    _, ypred = torch.max(torch.tensor(yhat), 1)
    ypred = ypred.cpu().detach().numpy()
    fig_cm = confusion_matrix_plot.plot_confusion_matrix_from_data(ytrue,
                                                                   ypred,
                                                                   ['Negative',
                                                                    'Positive'])
    wandb.log({dataset + " Confusion Matrix": wandb.Image(fig_cm)})


# apply threshold to positive probabilities to create labels
def to_labels(pos_probs, threshold):
    return (pos_probs >= threshold).astype('int')


def get_pr_auc(yhat, ytrue):
    ytrue = ytrue.cpu().detach().numpy()
    # retrieve just the probabilities for the positive class
    pos_probs = yhat[:, 1]
    # get predicted labels
    _, ypred = torch.max(torch.tensor(yhat), 1)
    ypred = ypred.cpu().detach().numpy()
    # predict class values
    precision, recall, thresholds = precision_recall_curve(ytrue, pos_probs)
    f1, pr_auc = f1_score(ytrue, ypred), auc(recall, precision)

    return precision, recall, f1, pr_auc, thresholds


def plot_precision_recall(model, yhat, ytrue, dataset='Test'):
    model.eval()
    with torch.no_grad():
        precision, recall, f1, pr_auc, thresholds = get_pr_auc(yhat, ytrue)
        # convert to f score
        fscore = (2 * precision * recall) / (precision + recall)
        fscore = np.nan_to_num(fscore)
        # locate the index of the largest f score
        ix = np.argmax(fscore)
        print('Best Threshold=%f, F-Score=%.3f' % (thresholds[ix], fscore[ix]))



        # print('Best Threshold=%f, F-Score=%.3f' % (thresholds[ix], scores[ix]))

        # summarize scores
        print(dataset + ' TCN: f1=%.3f pr_auc=%.3f' % (f1, pr_auc))
        wandb.log({'Model_' + dataset + '_PR_AUC': pr_auc,
                   'Model_' + dataset+'_F1': f1})

        '''
            Plot PR Curve
        '''
        fig = plt.figure()
        # plot the precision-recall curves
        no_skill = len(ytrue[ytrue == 1]) / len(ytrue)
        plt.plot([0, 1], [no_skill, no_skill], linestyle='--',
                 label='No Skill', zorder=1)
        plt.plot(recall, precision, marker='.', label='AUC:'+str(round(
            pr_auc, 3)), zorder=2)
        plt.scatter(recall[ix], precision[ix], marker='o', color='black',
                    label='Best', zorder=3)
        # axis labels
        plt.title(dataset+' Precision-Recall Curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        # show the legend
        plt.legend()
        # show the plot
        fig.show()
        wandb.log({dataset+' PR Curve': wandb.Image(fig)})

        return precision, recall, f1, pr_auc


def get_metrics_threshold(yhat, ytrue):
    probs = yhat[:, 1]
    # define thresholds
    thresholds = np.arange(0, 1, 0.01)
    N = len(thresholds)
    tn = [None] * N
    fp = [None] * N
    fn = [None] * N
    tp = [None] * N
    recall = [None] * N
    precision = [None] * N
    accuracy = [None] * N
    bacc = [None] * N
    tss = [None] * N
    hss = [None] * N
    mcc = [None] * N

    # evaluate each threshold
    cm = [sklearn.metrics.confusion_matrix(ytrue, to_labels(probs, t))
          for t in thresholds]
    for p in range(len(thresholds)):
        recall[p], precision[p], accuracy[p], bacc[p], tss[p], hss[p], tp[p], \
        fn[p], fp[p], tn[p], mcc[p] = calculate_metrics(cm[p], 2)

    # get best threshold
    ix = np.argmax(tss)
    print('Best Threshold=%f, TSS=%.3f' % (thresholds[ix], tss[ix]))
    return recall[ix], precision[ix], accuracy[ix], bacc[ix], tss[ix], hss[
        ix], tp[ix], fn[ix], fp[ix], tn[ix], thresholds[ix]
