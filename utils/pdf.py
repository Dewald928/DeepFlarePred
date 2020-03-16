# example of parametric probability density estimation
from matplotlib import pyplot
from numpy import asarray
from numpy import exp
from numpy import mean
from numpy import std
from numpy.random import normal
from scipy.stats import norm
from sklearn.neighbors import KernelDensity
import seaborn as sns
from scipy import stats
import wandb


def plot_density_estimation(data_loader, labels, dataset_name):
    # seaborn prob dens plot
    fig = plt.figure()
    yhat = infer_model(model, device, data_loader, args)
    x = yhat[:, 1]
    y = labels
    x_flare = x[np.where(y >= 1)[0]]
    x_no_flare = x[np.where(y < 1)[0]]
    sns.distplot(x_flare, bins=20)
    sns.distplot(x_no_flare, bins=20)
    # axis labels
    plt.title('Probability Density Estimation')
    plt.xlabel('Threshold')
    plt.ylabel('Probability Estimate')
    # plt.show()
    fig.show()
    wandb.log({dataset_name+' Probability Density Plot': wandb.Image(fig)})
