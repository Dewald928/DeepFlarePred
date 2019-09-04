import time
import copy
import torch


class Trainer():
    def __init__(self, model, criterion, optimizer, scheduler, dataloader, device, args):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.dataloader = dataloader
        self.device = device
        self.args = args
        self.epochs = self.args.epochs

        self.train_len = len(dataloader.y_train_data)
        self.val_len = len(dataloader.y_test_data)

        self.dataset_sizes = {
            'train': self.train_len,
            'val': self.val_len,
        }

        self.dataloaders_cat = {
            'train': self.dataloader.train_loader,
            'val': self.dataloader.valid_loader,
            # 'test': test_loader
        }

    def train_model(self):
        since = time.time()
        use_cuda = True

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0

        for epoch in range(self.epochs):
            print('Epoch {}/{}'.format(epoch, self.epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    # scheduler.step()  # comment out to disable scheduler
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for i, (inputs, labels) in enumerate(self.dataloaders_cat[phase]):
                    correct = 0
                    total = 0
                    # Load inputs as a torch tensor with gradient accumulation abilities
                    inputs = inputs.view(-1, self.model.series_len, self.model.n_features).requires_grad_()  # seq_dim

                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = self.criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / self.dataset_sizes[phase]
                epoch_acc = running_corrects.double() / self.dataset_sizes[phase]

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

                if phase == 'train':
                    train_acc = float(epoch_acc)
                else:
                    val_acc = float(epoch_acc)

                    # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    selected_epoch = epoch
                    best_model_wts = copy.deepcopy(self.model.state_dict())

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))
        #     print('Selected Model @Epoch:', selected_epoch)

        # load best model weights
        # model.load_state_dict(best_model_wts)     #comment out if not the best valid set is chosen
