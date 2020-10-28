import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import pandas as pd

import torch

from captum.attr import *

import shap
from utils import math_stuff

import wandb


def interpret_model(model, input_df, backgroud_df, device='cpu'):
    print("\n Interpreting Model...")
    is_seq = True if len(input_df.shape)>2 else False
    input_tensor = torch.tensor(input_df).float()
    background = torch.tensor(backgroud_df).float()
    input_tensor.requires_grad = True
    # l0 = background.shape[0]
    # l1 = input_tensor.shape[0]
    # input_tensor = input_tensor[l1-l0:,:]
    model.eval()

    model = model.to(device)
    input_tensor = input_tensor.to(device)

    sal = Saliency(model)
    ig = IntegratedGradients(model)
    dl = DeepLift(model)
    input_x_gradient = InputXGradient(model)
    gbp = GuidedBackprop(model)
    # occ = Occlusion(model)
    abl = FeatureAblation(model)
    svs = ShapleyValueSampling(model)

    attr_sal = sal.attribute(input_tensor, target=1)
    attr_ig, delta_ig = ig.attribute(input_tensor, target=1,
                                     return_convergence_delta=True)
    attr_dl, delta_dl = dl.attribute(input_tensor, target=1,
                                     return_convergence_delta=True)
    attr_ixg = input_x_gradient.attribute(input_tensor, target=1)
    attr_gbp = gbp.attribute(input_tensor, target=1)
    # attr_occ = occ.attribute(input_tensor, target=1,
    #                          sliding_window_shapes=(1,))
    attr_abl = abl.attribute(input_tensor, target=1)
    attr_shap = svs.attribute(input_tensor, target=1)

    return [attr_sal, attr_ig, attr_dl, attr_ixg, attr_gbp, attr_abl,
            attr_shap]


# Helper method to print importance and visualize distribution
def visualize_importance(feature_names, attr, n_features,
                         title="Average Feature Importance", plot=True):
    importance_avg = np.mean(attr.detach().numpy(), axis=0)
    importance_std = np.std(attr.detach().numpy(), axis=0)
    if importance_avg.shape.__len__() == 1:  # seq len 1 reshape
        importance_avg = np.reshape(importance_avg, (-1, 1))
        importance_std = np.reshape(importance_std, (-1, 1))
    importance_avg = importance_avg[:, -1] if importance_avg.shape[
                                                  1] > 0 else importance_avg
    importance_std = importance_std[:, -1] if importance_std.shape[
                                                  1] > 0 else importance_std

    x_pos = (np.arange(len(feature_names)))
    if plot:
        fig = plt.figure(figsize=(8, 4))
        plt.bar(x_pos, importance_avg, yerr=importance_std, align='center')
        plt.xticks(x_pos, feature_names, rotation='vertical')
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.title(title)
        plt.tight_layout()
        fig.show()
        wandb.log({f"Feature Importance: {title}": wandb.Image(fig)})

        # plot ranked
        df_sorted = pd.DataFrame(
            {'Features': feature_names, "Importances": importance_avg})
        df_sorted = df_sorted.sort_values('Importances', ascending=False)
        fig_sorted = df_sorted.plot(kind='bar', y='Importances', x='Features',
                                    title=title, figsize=(8, 4),
                                    yerr=importance_std)
        plt.ylabel('Importance')
        plt.tight_layout()
        plt.show()
        wandb.log(
            {f"Sorted Feature Importance: {title}": wandb.Image(fig_sorted)})


def plot_all_attr(attrs_list, feature_list, attr_name_list):
    n0, n1 = math_stuff.get_largest_primes(len(feature_list))
    fig, axes = plt.subplots(n0, n1, figsize=(20, 20), sharex=True,
                             sharey=True)
    axes = axes.reshape(-1)
    for i, feature in enumerate(feature_list):
        axes[i].set(title=feature)
        for j, attr in enumerate(attrs_list):
            # print(attr.shape) #todo is TCN compatible?
            importance_avg = np.mean(attr[:,i].detach().numpy(), axis=0)
            importance_std = np.std(attr[:,i].detach().numpy(), axis=0)
            # if importance_avg.shape.__len__() == 1:  # seq len 1 reshape
            #     importance_avg = np.reshape(importance_avg, (-1, 1))
            #     importance_std = np.reshape(importance_std, (-1, 1))
            # importance_avg = importance_avg[:, -1] if importance_avg.shape[1] > 0 else importance_avg
            # importance_std = importance_std[:, -1] if importance_std.shape[ 1] > 0 else importance_std

            axes[i].barh(attr_name_list[j], importance_avg,
                         xerr=importance_std,
                         align='center', label=attr_name_list[j])
    plt.show()


def plot_attr_vs_time(attrs_list, feature_list, attr_name_list):
    n0, n1 = math_stuff.get_largest_primes(len(feature_list))
    fig, axes = plt.subplots(n0, n1, figsize=(20, 20), sharex=True)
    axes = axes.reshape(-1)
    for i, feature in enumerate(feature_list):
        axes[i].set(title=feature)
        for j, attr in enumerate(attrs_list):
            # print(attr.shape) # todo TCN?
            attr = attr if len(attr.shape) == 2 else attr[:, :, -1]
            importance = attr[:,i].cpu().detach().numpy()
            axes[i].plot(importance, label=attr_name_list[j])
            axes[i].axvspan(xmin=128,
                            xmax=152, ymin=0, ymax=1,
                            alpha=0.1, color='r')
    plt.legend()
    wandb.log({'Attribution over Time': wandb.Image(fig)})
    plt.show()


def log_attrs(attrs_list, feature_list, attr_name_list, cfg):

    for i, attr_name in enumerate(attr_name_list):
        attrs_list[i] = attrs_list[i] if len(attrs_list[i].shape) == 2 else \
            attrs_list[i][:, :, -1]
        df_attr = pd.DataFrame(attrs_list[i].cpu().detach().numpy(),
                               columns=feature_list)

        df_attr.to_csv(
            f"./saved/results/attribution/{cfg.model_type}"
            f"/{attr_name.replace(' ', '')}"
            f"_{cfg.seed}.csv", index=False)
    print('ja man lekker logs')


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
        test_numpy = test_numpy[:, :, -1]
        for i in shap_values:
            shap_numpy.append(i[:, :, -1])

    # plot shap values
    fig_shap = plt.figure()
    plt.title('SHAP Summary Plot')
    shap.summary_plot(shap_numpy[1], test_numpy, feature_names=feature_names,
                      max_display=cfg.n_features)
    plt.tight_layout()
    fig_shap.show()
    wandb.log({'SHAP Summary Plot': wandb.Image(fig_shap)})
    # plt.close(fig_shap)

    # for single sample 151, X9.3 flare
    fig = shap.force_plot(e.expected_value[0], shap_numpy[1][151],
                          matplotlib=True, feature_names=feature_names,
                          link='identity', show=False)
    fig_shap1 = plt.gcf()
    # plt.title('SHAP Force Plot')
    plt.tight_layout()
    fig_shap1.show()
    wandb.log({'SHAP Force Plot': wandb.Image(fig_shap1)})
