import numpy as np
import sklearn
import torch
from matplotlib import pyplot

# precision-recall curve and f1
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from matplotlib import pyplot


def calculate_metrics(confusion_matrix, nclass):
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
        bacc[p] = round(0.5 * (
                float(tp[p]) / float(tp[p] + fn[p]) + float(tn[p]) / float(
            tn[p] + fp[p])), 3)
        hss[p] = round(2 * float(tp[p] * tn[p] - fp[p] * fn[p]) / float(
            (tp[p] + fn[p]) * (fn[p] + tn[p]) + (tp[p] + fp[p]) * (
                    fp[p] + tn[p])), 3)
        tss[p] = round((float(tp[p]) / float(tp[p] + fn[p] + 1e-6) - float(
            fp[p]) / float(fp[p] + tn[p] + 1e-6)), 3)

    print("tss: " + str(tss))
    print("hss: " + str(hss))
    print("bacc: " + str(bacc))
    print("accuracy: " + str(accuracy))
    print("precision: " + str(precision))
    print("recall: " + str(recall))

    return recall, precision, accuracy, bacc, tss, hss, tp, fn, fp, tn


def get_roc(model, dataloader, ytrue, device):
    # calculate roc curve
    yhat = model(dataloader.dataset.data.to(device))
    # retrieve just the probabilities for the positive class
    pos_probs = yhat[:, 1]
    # get predictied labels
    _, ypred = torch.max(yhat.data, 1)
    # plot no skill roc curve
    pyplot.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
    # calculate roc curve for model
    fpr, tpr, _ = sklearn.metrics.roc_curve(
        ytrue.cpu().detach().numpy(),
        pos_probs.cpu().detach().numpy())
    # calculate roc auc
    roc_auc = sklearn.metrics.roc_auc_score(ytrue.cpu().detach().numpy(),
                                            pos_probs.cpu().detach().numpy())
    print('Model ROC AUC %.3f' % roc_auc)
    # plot model roc curve
    pyplot.plot(fpr, tpr, marker='.', label='TCN')
    # axis labels
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    # show the legend
    pyplot.legend()
    # show the plot
    pyplot.show()


def get_precision_recall(model, dataloader, ytrue, device):
    # calculate roc curve
    ytrue = ytrue.cpu().detach().numpy()
    yhat = model(dataloader.dataset.data.to(device))
    # yhat = yhat.cpu().detach().numpy()
    # retrieve just the probabilities for the positive class
    pos_probs = yhat[:, 1]
    pos_probs = pos_probs.cpu().detach().numpy()
    # get predictied labels
    _, ypred = torch.max(yhat.data, 1)
    ypred = ypred.cpu().detach().numpy()
    # predict class values
    lr_precision, lr_recall, _ = precision_recall_curve(ytrue, pos_probs)
    lr_f1, lr_auc = f1_score(ytrue, ypred), auc(lr_recall, lr_precision)
    # summarize scores
    print('TCN: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))
    # plot the precision-recall curves
    no_skill = len(ytrue[ytrue == 1]) / len(ytrue)
    pyplot.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
    pyplot.plot(lr_recall, lr_precision, marker='.', label='Logistic')
    # axis labels
    pyplot.xlabel('Recall')
    pyplot.ylabel('Precision')
    # show the legend
    pyplot.legend()
    # show the plot
    pyplot.show()

