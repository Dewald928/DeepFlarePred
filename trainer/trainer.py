import time
import copy
import torch
import wandb
from torchsummary import summary

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

    def train(self, args, epoch):
        self.model.train()
        train_loss = 0
        correct = 0

        for batch_idx, (inputs, target) in enumerate(self.dataloader.train_loader):
            inputs = inputs.view(-1, self.model.series_len, self.model.n_features).requires_grad_()  # seq_dim
            inputs, target = inputs.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(inputs)
            loss = self.criterion(output, target)

            a = list(self.model.parameters())[0].clone()
            loss.backward()
            self.optimizer.step()
            b = list(self.model.parameters())[0].clone()
            torch.equal(a.data, b.data)

            # sum up batch loss
            train_loss += loss.item()
            # get the index of the max log-probability
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(inputs), len(self.dataloader.train_loader.dataset),
                           100. * batch_idx / len(self.dataloader.train_loader), loss.item()))
        train_loss /= len(self.dataloader.train_loader.dataset)
        wandb.log({
            "Train_Accuracy": 100. * correct / len(self.dataloader.train_loader.dataset),
            "Train_Loss": train_loss})

    def validate(self, args):
        self.model.eval()
        test_loss = 0
        correct = 0

        example_images = []
        with torch.no_grad():
            for inputs, target in self.dataloader.valid_loader:
                inputs = inputs.view(-1, self.model.series_len, self.model.n_features).requires_grad_()  # seq_dim
                inputs, target = inputs.to(self.device), target.to(self.device)
                output = self.model(inputs)
                # sum up batch loss
                test_loss += self.criterion(output, target).item()
                # get the index of the max log-probability
                pred = output.max(1, keepdim=True)[1]
                correct += pred.eq(target.view_as(pred)).sum().item()
                example_images.append(wandb.Image(
                    inputs[0], caption="Pred: {} Truth: {}".format(pred[0].item(), target[0])))

        test_loss /= len(self.dataloader.valid_loader.dataset)
        print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(self.dataloader.valid_loader.dataset),
            100. * correct / len(self.dataloader.valid_loader.dataset)))
        wandb.log({
            "Examples": example_images,
            "Validation_Accuracy": 100. * correct / len(self.dataloader.valid_loader.dataset),
            "Validation_Loss": test_loss})

    def train_model(self):
        since = time.time()

        for epoch in range(1, self.args.epochs + 1):
            self.train(self.args,  epoch)
            self.validate(self.args)


        # use_cuda = True
        #
        # best_model_wts = copy.deepcopy(self.model.state_dict())
        # best_acc = 0.0
        #
        # for epoch in range(self.epochs):
        #     print('Epoch {}/{}'.format(epoch, self.epochs - 1))
        #     print('-' * 10)
        #
        #     # Each epoch has a training and validation phase
        #     for phase in ['train', 'val']:
        #         if phase == 'train':
        #             # scheduler.step()  # comment out to disable scheduler
        #             self.model.train()  # Set model to training mode
        #         else:
        #             self.model.eval()  # Set model to evaluate mode
        #
        #         running_loss = 0.0
        #         running_corrects = 0
        #
        #         # Iterate over data.
        #         for i, (inputs, labels) in enumerate(self.dataloaders_cat[phase]):
        #             correct = 0
        #             total = 0
        #             # Load inputs as a torch tensor with gradient accumulation abilities
        #             inputs = inputs.view(-1, self.model.series_len, self.model.n_features).requires_grad_()  # seq_dim
        #
        #             # zero the parameter gradients
        #             self.optimizer.zero_grad()
        #
        #             # forward
        #             # track history if only in train
        #             with torch.set_grad_enabled(phase == 'train'):
        #                 outputs = self.model(inputs)
        #                 _, preds = torch.max(outputs, 1)
        #                 loss = self.criterion(outputs, labels)
        #
        #                 # backward + optimize only if in training phase
        #                 if phase == 'train':
        #                     loss.backward()
        #                     self.optimizer.step()
        #
        #             # statistics
        #             running_loss += loss.item() * inputs.size(0)
        #             running_corrects += torch.sum(preds == labels.data)
        #
        #         epoch_loss = running_loss / self.dataset_sizes[phase]
        #         epoch_acc = running_corrects.double() / self.dataset_sizes[phase]
        #
        #         print('{} Loss: {:.4f} Acc: {:.4f}'.format(
        #             phase, epoch_loss, epoch_acc))
        #
        #         if phase == 'train':
        #             train_acc = float(epoch_acc)
        #         else:
        #             val_acc = float(epoch_acc)
        #
        #             # deep copy the model
        #         if phase == 'val' and epoch_acc > best_acc:
        #             best_acc = epoch_acc
        #             selected_epoch = epoch
        #             best_model_wts = copy.deepcopy(self.model.state_dict())

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        # print('Best val Acc: {:4f}'.format(best_acc))
        #     print('Selected Model @Epoch:', selected_epoch)

        # load best model weights
        # model.load_state_dict(best_model_wts)     #comment out if not the best valid set is chosen
