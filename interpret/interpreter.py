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

import shap

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

            ig = IntegratedGradients(model.to(device))
            sal = Saliency(model.to(device))  # todo deeplift not working
            # temp_data = torch.clone(data).to(device)
            attr_ig = ig.attribute(data, target=1)
            attr_sal = sal.attribute(inputs=data, target=1)
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
        plt.xticks(x_pos, feature_names, rotation='vertical')
        plt.xlabel(axis_title)
        plt.title(title)
        plt.tight_layout()
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
        plt.tight_layout()
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


def get_shap(model, test_loader, device, args, feature_names, start_feature):
    batch = next(iter(test_loader))
    samples, _ = batch
    print(samples.size())

    test_sample_x = test_loader.dataset.data[4982:4996].to(
        device)  # 4940 - 5121
    background = samples[:100].to(device)
    # test_samples = samples[101:200].to(device)
    test_samples = test_sample_x
    e = shap.DeepExplainer(model, background)
    shap_values = e.shap_values(test_samples)
    # shap_numpy = shap_values  # comment out for tcn
    # uncomment for tcn
    shap_numpy = []
    test_numpy = np.swapaxes(np.swapaxes(test_samples.cpu().numpy(), 1, -1), 1,
                             2)
    test_numpy = test_numpy.squeeze(2)
    for i in shap_values:
        shap_numpy.append(i.squeeze(2))

    fig_shap = plt.figure()
    plt.title('SHAP Summary Plot')
    plt.tight_layout()
    shap.summary_plot(shap_numpy[1], test_numpy,
                      feature_names=feature_names[start_feature:start_feature +
                                                                args.n_features],
                      max_display=args.n_features)
    fig_shap.show()
    wandb.log({'SHAP Summary Plot': wandb.Image(fig_shap)})
    # plt.close(fig_shap)

    # for single sample
    fig = shap.force_plot(e.expected_value[0], shap_numpy[1][0],
                          matplotlib=True, feature_names=feature_names[
                                                         start_feature:start_feature + args.n_features],
                          link='logit', show=False)
    fig_shap1 = plt.gcf()
    # plt.title('SHAP Force Plot')
    plt.tight_layout()
    fig_shap1.show()
    wandb.log({'SHAP Force Plot': wandb.Image(fig_shap1)})
