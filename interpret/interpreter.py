import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import pandas as pd

import torch

from captum.attr import IntegratedGradients
from captum.attr import DeepLift
from captum.attr import DeepLiftShap
from captum.attr import Saliency
from captum.attr import GradientAttribution

import wandb


def interpret_model(model, device, test_loader, n_features, args):
    print("\n Interpreting Model...")
    attr_ig = []
    attr_ig_avg = []
    attr_sal = []
    attr_sal_avg = []
    model.eval()
    i = 1
    with torch.no_grad():
        model.to(device)
        for data, target in test_loader:
            print("Batch#" + str(i))
            data, target = data.to(device), target.to(device)
            data = data.view(len(data), n_features, args.layer_dim)

            ig = IntegratedGradients(model.to(device))
            sal = Saliency(model.to(device))  # todo deeplift not working
            temp_data = torch.clone(data).to(device)
            attr_ig = ig.attribute(temp_data, target=1)
            attr_sal = sal.attribute(inputs=temp_data, target=1)
            try:
                attr_ig_avg = attr_ig if i == 1 else (
                                                             attr_ig_avg +
                                                             attr_ig) / i
                attr_sal_avg = attr_sal if i == 1 else (
                                                               attr_sal_avg
                                                               + attr_sal) / i
            except:
                print("hmmmm???")
                pass
            attr_ig = attr_ig.cpu().detach().numpy()
            attr_sal = attr_sal.cpu().detach().numpy()

            if i == len(test_loader) - 1:
                break
            i += 1
    attr_ig_avg = attr_ig_avg.cpu().detach().numpy()
    attr_sal_avg = attr_sal_avg.cpu().detach().numpy()

    return attr_ig, attr_sal, attr_ig_avg, attr_sal_avg


# Helper method to print importance and visualize distribution
def visualize_importance(feature_names, importances, std, n_features,
                         title="Average Feature Importance", plot=True,
                         axis_title="Features"):
    # print(title)
    # for i in range(len(feature_names)):
    #     print(feature_names[i], ": ", '%.3f' % (importances[i]))
    x_pos = (np.arange(len(feature_names)))
    if plot:
        fig = plt.figure(figsize=(8, 4))
        plt.bar(x_pos, importances.reshape(n_features),
                yerr=std.reshape(n_features), align='center')
        plt.xticks(x_pos, feature_names, wrap=False, rotation=60)
        plt.xlabel(axis_title)
        plt.title(title)
        fig.show()
        wandb.log({'Image of features': wandb.Image(fig)})

        # plot ranked
        df_sorted = pd.DataFrame({'Features': feature_names,
                                  "Importances": importances.reshape(
                                      n_features)})
        df_sorted = df_sorted.sort_values('Importances')
        fig_sorted = df_sorted.plot(kind='bar', y='Importances', x='Features',
                                    title=title, figsize=(8, 4),
                                    yerr=std.reshape(n_features))
        plt.show()
        wandb.log({'Feature Ranking': wandb.Image(fig_sorted)})


def check_significance(attr, test_features, feature_num=1):
    fig = plt.hist(attr[:, feature_num], 100)
    plt.title("Distribution of Attribution Values")
    plt.show()

    bin_means, bin_edges, _ = stats.binned_statistic(test_features[:, 1],
                                                     attr[:, 1],
                                                     statistic='mean', bins=6)
    bin_count, _, _ = stats.binned_statistic(test_features[:, 1], attr[:, 1],
                                             statistic='count', bins=6)

    bin_width = (bin_edges[1] - bin_edges[0])
    bin_centers = bin_edges[1:] - bin_width / 2
    plt.scatter(bin_centers, bin_means, s=bin_count)
    plt.xlabel("Average Sibsp Feature Value")
    plt.ylabel("Average Attribution")