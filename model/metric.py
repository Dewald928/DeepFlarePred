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


def get_roc(model, yhat, ytrue, device):
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
        # plot no skill roc curve
        plt.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
        # calculate roc curve for model
        fpr, tpr, _ = roc_curve(ytrue, pos_probs)
        # calculate roc auc
        roc_auc = roc_auc_score(ytrue, pos_probs)
        print('Model ROC AUC %.3f' % roc_auc)

        # plot model roc curve
        plt.plot(fpr, tpr, marker='.', label='TCN')
        # axis labels
        plt.title('ROC curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        # show the legend
        plt.legend()
        # show the plot
        fig.show()
        wandb.log({'ROC curve': fig, 'ROC_AUC': roc_auc})

        return roc_auc


def get_precision_recall(model, yhat, ytrue, device):
    model.eval()
    with torch.no_grad():
        # calculate roc curve
        ytrue = ytrue.cpu().detach().numpy()
        # retrieve just the probabilities for the positive class
        pos_probs = yhat[:, 1]
        # get predictied labels
        _, ypred = torch.max(torch.tensor(yhat), 1)
        ypred = ypred.cpu().detach().numpy()
        # predict class values
        precision, recall, _ = precision_recall_curve(ytrue, pos_probs)
        f1, pr_auc = f1_score(ytrue, ypred), auc(recall, precision)
        # summarize scores
        print('TCN: f1=%.3f pr_auc=%.3f' % (f1, pr_auc))
        wandb.log({'PR_AUC': pr_auc})

        '''
            Plot PR Curve
        '''
        fig = plt.figure()
        # plot the precision-recall curves
        no_skill = len(ytrue[ytrue == 1]) / len(ytrue)
        plt.plot([0, 1], [no_skill, no_skill], linestyle='--',
                 label='No Skill')
        plt.plot(recall, precision, marker='.', label='Model')
        # axis labels
        plt.title('Precision-Recall Curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        # show the legend
        plt.legend()
        # show the plot
        fig.show()
        wandb.log({'PR Curve': fig})

        return precision, recall, f1, pr_auc



