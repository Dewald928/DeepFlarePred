import time
import copy
import torch


class Trainer():
    def __init__(self, model, criterion, optimizer, scheduler, dataloaders, device, args):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.dataloaders = dataloaders
        self.device = device
        self.args = args
        self.epochs = self.args.epochs




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
                for i, (inputs, labels) in enumerate(self.dataloaders[phase]):
                    if use_cuda and torch.cuda.is_available():
                        inputs = inputs.reshape(-1, 28 * 28).to(self.device)  # remove reshape if cnn
                        # inputs = inputs.to(device)
                        labels = labels.to(self.device)

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

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

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
                    best_model_wts = copy.deepcopy(model.state_dict())

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))
        #     print('Selected Model @Epoch:', selected_epoch)

        # load best model weights
        # model.load_state_dict(best_model_wts)     #comment out if not the best valid set is chosen
        return model, train_acc, val_acc