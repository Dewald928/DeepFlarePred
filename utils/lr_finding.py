# doesn't work
import matplotlib.pyplot as plt
# from torch_lr_finder import LRFinder
from utils.lr_finder_with_metric import LRFinder
import wandb
import numpy as np


def get_prev_max(arr):
    ''' gets array from previous maximum loss before final minimum'''
    current = 0
    previous = arr[-1]
    i = 2
    c = 0
    while (c <= 1) and (i < len(arr)):
        previous = current
        current = arr[-i]
        i += 1
        if current < previous:
            c += 1
        # else:
        #     c = 0
    return i-1


def find_lr(model, optimizer, criterion, device, train_loader, valid_loader,
            cfg):
    num_iter = 100
    fig, axes = plt.subplots(2,2, figsize=(10,6), sharex=True)
    fig.set_tight_layout(True)
    lr_finder = LRFinder(model, optimizer, criterion, device=device,
                         metric_name='TSS')
    # lr_finder = LRFinder(model, optimizer, criterion, device=device)
    lr_finder.range_test(train_loader, end_lr=10, num_iter=num_iter)
    trainx = lr_finder.history['lr']
    trainy_TSS = lr_finder.history['TSS']
    trainy_loss = lr_finder.history['loss']

    # plot
    lr_finder.plot(ax=axes[0][0], skip_start=0, skip_end=0,
                   metric_name="TSS", show_lr=trainx[np.argmax(trainy_TSS)])
    lr_finder.plot(ax=axes[1][0], skip_start=0, skip_end=0,
                   metric_name="loss", show_lr=trainx[np.argmin(trainy_loss)])
    lr_finder.reset()  # to reset the model and optimizer to their initial state

    # halfway down slope for static
    # min max for cyclic

    # LR finder (Leslie SMiths)
    lr_finder = LRFinder(model, optimizer, criterion, device=device,
                         metric_name='TSS')
    # lr_finder = LRFinder(model, optimizer, criterion, device=device)
    lr_finder.range_test(train_loader, val_loader=valid_loader, end_lr=10,
                         num_iter=num_iter, step_mode="exp")

    valx = lr_finder.history['lr']
    valy_TSS = lr_finder.history['TSS']
    valy_loss = lr_finder.history['loss']

    lr_finder.plot(ax=axes[0][1], skip_end=0, skip_start=0,
                   metric_name="TSS", show_lr=valx[np.argmax(valy_TSS)])
    lr_finder.plot(ax=axes[1][1], skip_end=0, skip_start=0,
                   metric_name="loss", show_lr=valx[np.argmin(valy_loss)])
    # lr_finder.plot(ax=axes[1][1], skip_end=0, skip_start=0, show_lr=valx[np.argmin(valy_loss)])
    lr_finder.reset()

    trainx = np.pad(trainx, (0, num_iter - len(trainx)), 'constant',
                    constant_values=np.nan)
    trainy_loss = np.pad(trainy_loss, (0, num_iter - len(trainy_loss)),
                         'constant', constant_values=np.nan)
    trainy_TSS = np.pad(trainy_TSS, (0, num_iter - len(trainy_TSS)),
                        'constant', constant_values=np.nan)
    valx = np.pad(valx, (0, num_iter - len(valx)), 'constant',
                  constant_values=np.nan)
    valy_loss = np.pad(valy_loss, (0, num_iter - len(valy_loss)), 'constant',
                       constant_values=np.nan)
    valy_TSS = np.pad(valy_TSS, (0, num_iter - len(valy_TSS)), 'constant',
                      constant_values=np.nan)

    fig.tight_layout()
    plt.tight_layout()
    axes[0][0].set_title('Training')
    axes[1][0].set_title('Training')
    axes[0][1].set_title('Validation')
    axes[1][1].set_title('Validation')
    for ax in axes.flat:
        ax.set_ylim([0,1])

    if cfg.log_lr:
        for i in range(num_iter - 1):
            wandb.log(
                {'train_lr_TSS': trainy_TSS[i], 'train_lr_loss': trainy_loss[i],
                 'train_lr_step': trainx[i], 'valid_lr_TSS': valy_TSS[i],
                 'valid_lr_loss': valy_loss[i], 'valid_lr_step': valx[i]}, step=i)

    wandb.log({'LR_Finder_img': wandb.Image(fig)})
    plt.show()


    # maxlr, halwaylr on loss
    if cfg.lr_metric == 'Loss':
        print('Loss')
        valy_loss = np.array(lr_finder.history['loss'])
        valx = np.array(lr_finder.history['lr'])
        max_lr = valx[np.argmin(valy_loss)]
        # calculate gradient of loss
        valy_loss_g = np.gradient(valy_loss)
        # get turning point index
        sign = np.sign(valy_loss_g)
        turning_points = np.diff(sign, axis=0)
        tp_idx = np.argwhere(np.abs(turning_points)==2).reshape(-1)
        # last and before last turning point cut out
        before_min_loss = np.array(valy_loss[tp_idx[-2]:tp_idx[-1]])
        before_min_lr = np.array(valx[tp_idx[-2]:tp_idx[-1]])
        min_lr = before_min_lr[np.argmax(before_min_loss)]
        # min_lr = before_min_lr[-get_prev_max(before_min_loss)]
        halfway_lr = 0.5*(np.log(min_lr)-np.log(max_lr))
        halfway_lr = np.exp(np.log(max_lr) + halfway_lr)
    elif cfg.lr_metric == 'TSS':
        print("TSS")
        valy_TSS = lr_finder.history['TSS']
        valx = lr_finder.history['lr']
        max_lr = valx[np.argmax(valy_TSS)]
        before_max_TSS = valy_TSS[0:np.argmax(valy_TSS)]
        min_lr = valx[np.argmin(before_max_TSS)]
        halfway_lr = 0.5 * (np.log(min_lr) - np.log(max_lr))
        halfway_lr = np.exp(np.log(max_lr) + halfway_lr)
    else:
        print('Woopsie')

    return min_lr, halfway_lr, max_lr


def closest_arg(array, value):
    absolute_val_array = np.abs(array - value)
    smallest_difference_index = absolute_val_array.argmin()
    closest_element = array[smallest_difference_index]
    return smallest_difference_index





