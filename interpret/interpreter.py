import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import pandas as pd

import torch

from captum.attr import *

import shap

import wandb


def interpret_model(model, device, input_df, backgroud_df):
    print("\n Interpreting Model...")

    input_tensor = torch.tensor(input_df).float()
    background = torch.tensor(backgroud_df).float()
    input_tensor.requires_grad = True
    model.eval()

    model = model.to('cpu')
    input_tensor = input_tensor.to('cpu')

    sal = Saliency(model)
    ig = IntegratedGradients(model)
    dl = DeepLift(model)
    input_x_gradient = InputXGradient(model)
    gbp = GuidedBackprop(model)


    attr_sal = sal.attribute(input_tensor, target=1)
    attr_ig, delta_ig = ig.attribute(input_tensor, target=1,
                                     return_convergence_delta=True)
    attr_dl, delta_dl = dl.attribute(input_tensor, target=1,
                                     return_convergence_delta=True)
    attr_ixg = input_x_gradient.attribute(input_tensor, target=1)
    attr_gbp = gbp.attribute(input_tensor, target=1)

    return [attr_sal, attr_ig, delta_ig, attr_dl, delta_dl, attr_ixg, attr_gbp]


# Helper method to print importance and visualize distribution
def visualize_importance(feature_names, attr, n_features,
                         title="Average Feature Importance", plot=True):

    importance_avg = np.mean(attr.detach().numpy(), axis=0)
    importance_std = np.std(attr.detach().numpy(), axis=0)
    if importance_avg.shape.__len__() == 1:  # seq len 1 reshape
        importance_avg = np.reshape(importance_avg, (-1, 1))
        importance_std = np.reshape(importance_std, (-1, 1))
    importance_avg = importance_avg[:,-1] if importance_avg.shape[1] > 0 else importance_avg
    importance_std = importance_std[:,-1] if importance_std.shape[1] > 0 else importance_std

    x_pos = (np.arange(len(feature_names)))
    if plot:
        fig = plt.figure(figsize=(8, 4))
        plt.bar(x_pos, importance_avg,
                yerr=importance_std,
                align='center')
        plt.xticks(x_pos, feature_names, rotation='vertical')
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.title(title)
        plt.tight_layout()
        fig.show()
        wandb.log({f"Feature Importance: {title}": wandb.Image(fig)})

        # plot ranked
        df_sorted = pd.DataFrame({'Features': feature_names,
                                  "Importances": importance_avg})
        df_sorted = df_sorted.sort_values('Importances', ascending=False)
        fig_sorted = df_sorted.plot(kind='bar', y='Importances', x='Features',
                                    title=title, figsize=(8, 4),
                                    yerr=importance_std)
        plt.ylabel('Importance')
        plt.tight_layout()
        plt.show()
        wandb.log({f"Sorted Feature Importance: {title}": wandb.Image(
            fig_sorted)})


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


def get_shap(model, input_df, backgroud_df, device, cfg, feature_names,
             start_feature):

    test_samples = torch.tensor(input_df).float().to(device)
    background = torch.tensor(backgroud_df).float().to(device)

    e = shap.DeepExplainer(model.to(device), background)
    shap_values = e.shap_values(test_samples)

    if cfg.model_type == 'MLP':
        shap_numpy = shap_values
        test_numpy = test_samples.cpu().numpy()
    else:
        shap_numpy = []
        test_numpy = test_samples.cpu().numpy()
        test_numpy = test_numpy[:,:,-1]
        for i in shap_values:
            shap_numpy.append(i[:,:,-1])

    # plot shap values
    fig_shap = plt.figure()
    plt.title('SHAP Summary Plot')
    shap.summary_plot(shap_numpy[1], test_numpy,
                      feature_names=feature_names[start_feature:start_feature +
                                                                cfg.n_features],
                      max_display=cfg.n_features)
    plt.tight_layout()
    fig_shap.show()
    wandb.log({'SHAP Summary Plot': wandb.Image(fig_shap)})
    # plt.close(fig_shap)

    # for single sample 151, X9.3 flare
    fig = shap.force_plot(e.expected_value[0], shap_numpy[1][151],
                          matplotlib=True, feature_names=feature_names[
                                                         start_feature:start_feature + cfg.n_features],
                          link='identity', show=False)
    fig_shap1 = plt.gcf()
    # plt.title('SHAP Force Plot')
    plt.tight_layout()
    fig_shap1.show()
    wandb.log({'SHAP Force Plot': wandb.Image(fig_shap1)})
