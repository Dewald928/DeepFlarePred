# example of parametric probability density estimation
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import wandb


def plot_density_estimation(yhat, labels, dataset_name):
    # seaborn prob dens plot
    fig = plt.figure()
    x = yhat[:, 1]
    y = labels
    x_flare = x[np.where(y >= 1)[0]]
    x_no_flare = x[np.where(y < 1)[0]]
    sns.distplot(x_flare, bins=20, color='r')
    sns.distplot(x_no_flare, bins=20, color='b')
    # axis labels
    plt.title(dataset_name + ' Probability Density Estimation')
    plt.xlabel('Threshold')
    plt.ylabel('Probability Estimate')
    plt.legend(['Flare', 'No_Flare'])
    # plt.show()
    fig.show()
    wandb.log({dataset_name+' Probability Density Plot': wandb.Image(fig)})
