# example of parametric probability density estimation
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import wandb
from scipy.stats import norm
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from model import metric
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve


def plot_density_estimation(net, yhat, labels, dataset_name):
    # seaborn prob dens plot
    fig = plt.figure(dpi=200)
    x = yhat
    y = labels
    x_flare = x[np.where(y >= 1)[0]]
    x_no_flare = x[np.where(y < 1)[0]]
    ax0 = sns.distplot(x_flare, bins=30, fit=norm,
                       kde_kws={'color': 'r', 'label': 'Flare_KDE'},
                       fit_kws={'color': '#ff81c0', 'label': 'Flare_Normal'},
                       hist_kws={'color': 'r'})
    ax1 = sns.distplot(x_no_flare, bins=30, fit=norm,
                       kde_kws={'color': 'b', 'label': 'No_Flare_KDE'},
                       fit_kws={'color': '#75bbfd',
                                'label': 'No_Flare_Normal'},
                       hist_kws={'color': 'b'})

    x = ax0.lines[3].get_xdata()
    y = ax0.lines[3].get_ydata()
    f = ax0.lines[3].get_ydata()
    g = ax0.lines[1].get_ydata()
    maxid = np.argmax(y)  # The id of the peak (maximum of y data)
    idx = np.argwhere(np.diff(np.sign(f - g))).flatten()  # intersection
    stdidx = maxid + int(round(0.341 * (len(x) - 1 - maxid)))
    # plt.plot(x[idx], y[idx], 'k|', ms=10)
    # plt.plot(x[stdidx], y[stdidx], 'b*', ms=10)

    # axis labels
    plt.title(dataset_name + ' Probability Density Estimation')
    plt.xlabel('Threshold Probability')
    plt.ylabel('Count')
    # plt.legend(['Flare', 'No_Flare'])
    # plt.show()
    plt.legend()
    fig.show()
    wandb.log({dataset_name + ' Probability Density Plot': wandb.Image(fig)})

    return x[idx]


def plot_calibration_curve(yprob, ytrue, ax, name='Test'):
    ax2 = ax.twinx()

    ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    # ax.plot([0, 1], [0, 0.0075], "k:", label="BSS=0")

    # fraction of positives
    fop, mean_predicted_value = calibration_curve(ytrue, yprob, n_bins=20,
                                                  normalize=True)
    bss = metric.bss_analysis(yprob, ytrue)
    ax.plot(mean_predicted_value, fop, "s-",
            label="%s (BSS: %1.1f)" % (name, bss), zorder=1)

    ax2.hist(yprob, range=(0, 1), bins=20, label=name, histtype="step",
             lw=2, color='r', zorder=0)
    ax.set_ylabel("Fraction of positives")
    ax.set_xlabel("Probability")
    ax.set_ylim([-0.05, 1.05])
    ax.legend(loc="upper right")
    ax.set(title='c)')
    # ax.set_title('Calibration plots  (reliability curve)')

    ax2.set_ylabel("Count")
    ax2.legend(loc="upper left", ncol=2)

    ax.grid(True)
    plt.tight_layout()
    # fig.show()
    wandb.log({name + ' Probability Calibration Curve': wandb.Image(plt)})


def plot_ssp(yprob, ytrue, ax):
    tss, hss, thresholds = metric.get_metrics_threshold(yprob, ytrue)[8:11]
    ax.plot(thresholds, tss, label='TSS')
    ax.plot(thresholds, hss, label='HSS')
    ax.legend()
    ax.set(xlabel='Threshold Probability')
    ax.set(ylabel='TSS, HSS')
    ax.set(xlim=(0, 1))
    ax.set(ylim=(0, 1))
    ax.set(title='a)')
    ax.grid(True)
    plt.tight_layout()


def plot_roc_curve(yprob, ytrue, ax):
    fpr, tpr, thresholds = roc_curve(ytrue, yprob)
    # calculate roc auc
    roc_auc = roc_auc_score(ytrue, yprob)
    # print('Model ROC AUC %.3f' % roc_auc)
    # get the best threshold

    J = tpr - fpr
    ix = np.argmax(J)
    # print('Best Threshold=%f, J stat=%.3f' % (thresholds[ix], J[ix]))
    ax.plot([0, 1], [0, 1], linestyle='--', label='No Skill', zorder=1)
    ax.plot(fpr, tpr, marker='.', label='ROC AUC: ' + str(round(roc_auc, 3)),
            zorder=2)
    ax.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Best TSS',
               zorder=3)
    # axis labels
    ax.set(xlabel='False Positive Rate')
    ax.set(ylabel='True Positive Rate')
    ax.set(ylim=(0, 1.01))
    ax.set(title='b)')
    # show the legend
    ax.legend()
    ax.grid(True)
    plt.tight_layout()


def plot_eval_graphs(yprob, ytrue, dataset='Test'):
    # plot Reliability diagram
    # plot ROC
    # plot thresholded metrics (SSP) skill score profiles
    fig, axes = plt.subplots(1, 3, figsize=(12, 5), sharey=True, dpi=200)
    plot_ssp(yprob, ytrue, axes[0])
    plot_roc_curve(yprob, ytrue, axes[1])
    plot_calibration_curve(yprob, ytrue, axes[2], dataset)
    plt.tight_layout()
    wandb.log({dataset+' Evaluation Plot': wandb.Image(fig)})
    fig.show()
