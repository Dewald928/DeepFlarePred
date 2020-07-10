# doesn't work
import matplotlib.pyplot as plt
from torch_lr_finder import LRFinder
import wandb


def find_lr(model, optimizer, criterion, device, train_loader, valid_loader):
    fig, axes = plt.subplots(1,2, figsize=(10,6))
    fig.set_tight_layout(True)
    lr_finder = LRFinder(model, optimizer, criterion, device=device)
    lr_finder.range_test(train_loader, end_lr=1, num_iter=1000)
    lr_finder.plot(ax=axes[0])  # to inspect the loss-learning rate graph
    lr_finder.reset()  # to reset the model and optimizer to their initial state
    # halfway down slope for static
    # min max for cyclic

    # LR finder (Leslie SMiths)
    lr_finder = LRFinder(model, optimizer, criterion, device=device)
    lr_finder.range_test(train_loader, val_loader=valid_loader, end_lr=1,
                         num_iter=1000, step_mode="exp")
    lr_finder.plot(log_lr=True, ax=axes[1])
    lr_finder.reset()

    fig.tight_layout()
    plt.tight_layout()
    axes[0].set_title('Training')
    axes[1].set_title('Validation')
    trainx = fig.axes[0].lines[0]._x
    trainy = fig.axes[0].lines[0]._y
    valx = fig.axes[1].lines[0]._x
    valy = fig.axes[1].lines[0]._y
    for i in range(len(trainx)):
        wandb.log({'train_lr':trainy[i], 'train_lr_step': trainx[i]}, step=i)
    for i in range(len(valx)):
        wandb.log({'valid_lr': valy[i], 'valid_lr_step': valx[i]}, step=i)
    wandb.log({'LR_Finder_img': wandb.Image(fig)})
    # plt.show()


