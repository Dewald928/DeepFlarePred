# doesn't work
import matplotlib.pyplot as plt
from torch_lr_finder import LRFinder
import wandb
import numpy as np


def find_lr(model, optimizer, criterion, device, train_loader, valid_loader):
    num_iter = 100
    fig, axes = plt.subplots(1,2, figsize=(10,6))
    fig.set_tight_layout(True)
    lr_finder = LRFinder(model, optimizer, criterion, device=device)
    lr_finder.range_test(train_loader, end_lr=10, num_iter=num_iter)
    lr_finder.plot(ax=axes[0], skip_start=0, skip_end=0)  # to inspect the
    # loss-learning rate graph
    lr_finder.reset()  # to reset the model and optimizer to their initial state
    trainx = lr_finder.history['lr']
    trainy = lr_finder.history['loss']
    # halfway down slope for static
    # min max for cyclic

    # LR finder (Leslie SMiths)
    lr_finder = LRFinder(model, optimizer, criterion, device=device)
    lr_finder.range_test(train_loader, val_loader=valid_loader, end_lr=10,
                         num_iter=num_iter, step_mode="exp")
    lr_finder.plot(ax=axes[1], skip_end=0, skip_start=0)
    lr_finder.reset()
    valx = lr_finder.history['lr']
    valy = lr_finder.history['loss']

    trainx = np.pad(trainx, (0, num_iter - len(trainx)), 'constant',
           constant_values=(np.nan))
    trainy = np.pad(trainy, (0, num_iter - len(trainy)), 'constant',
           constant_values=(np.nan))
    valx = np.pad(valx, (0, num_iter - len(valx)), 'constant',
           constant_values=(np.nan))
    valy = np.pad(valy, (0, num_iter - len(valy)), 'constant',
           constant_values=(np.nan))

    fig.tight_layout()
    plt.tight_layout()
    axes[0].set_title('Training')
    axes[1].set_title('Validation')

    for i in range(num_iter):
        wandb.log({'train_lr': trainy[i], 'train_lr_step': trainx[i],
                   'valid_lr': valy[i], 'valid_lr_step': valx[i]}, step=i)
    # for i in range(len(valx)):
    wandb.log({'LR_Finder_img': wandb.Image(fig)})
    plt.show()


