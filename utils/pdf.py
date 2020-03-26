# example of parametric probability density estimation
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import wandb
from scipy.stats import norm


def plot_density_estimation(yhat, labels, dataset_name):
    # seaborn prob dens plot
    fig = plt.figure()
    x = yhat[:, 1]
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
    stdidx = maxid + int(round(0.341*(len(x)-1-maxid)))
    plt.plot(x[idx], y[idx], 'k|', ms=10)
    # plt.plot(x[stdidx], y[stdidx], 'b*', ms=10)

    # axis labels
    plt.title(dataset_name + ' Probability Density Estimation')
    plt.xlabel('Threshold Probability')
    plt.ylabel('Count')
    # plt.legend(['Flare', 'No_Flare'])
    # plt.show()
    plt.legend()
    fig.show()
    wandb.log({dataset_name+' Probability Density Plot': wandb.Image(fig)})

    return x[idx]
