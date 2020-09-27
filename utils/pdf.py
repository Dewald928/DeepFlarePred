# example of parametric probability density estimation
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import wandb
from scipy.stats import norm
from sklearn.calibration import calibration_curve, CalibratedClassifierCV


def plot_density_estimation(net, yhat, labels, dataset_name):
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


def plot_calibration_curve(est, name, X_valid, y_valid, X_test, y_test,
                           y_pred):
    # plot calibration curve
    """Plot calibration curve for est w/o and with calibration. """
    # Calibrated with isotonic calibration
    isotonic = CalibratedClassifierCV(est, cv='prefit', method='isotonic')

    # Calibrated with sigmoid calibration
    sigmoid = CalibratedClassifierCV(est, cv='prefit', method='sigmoid')

    fig = plt.figure(2, figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))
    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

    # todo apply calibration (make a function)
    # Logistic regression with no calibration as baseline
    # lr = LogisticRegression(C=1.)
    # (isotonic, name + ' + Isotonic'),
    # (sigmoid, name + ' + Sigmoid')

    # for clf, name in [(est, name)]:
    #     clf.fit(X_valid, y_valid)
    #     y_pred = clf.predict(X_test)

    # todo predict proba, brier skill score?
    fraction_of_positives, mean_predicted_value = calibration_curve(y_test,
                                                                    y_pred[:,
                                                                    1],
                                                                    n_bins=20, normalize=True)
    ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
             label="%s" % (name))

    ax2.hist(y_pred[:, 1], range=(0, 1), bins=20, label=name,
             histtype="step", lw=2)
    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration plots  (reliability curve)')

    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)

    plt.tight_layout()
    fig.show()
    wandb.log(
        {name + ' Probability Calibration Curve': wandb.Image(fig)})