import torch.utils.data


class CustomDataset(torch.utils.data.Dataset):
    """Create a custom dataset, given a tensor of values and a tensor of
    labels."""
    def __init__(self, x, y, x_transform=None, y_transform=None):
        self.data = x
        self.targets = y
        self.x_transform = x_transform
        self.y_transform = y_transform

    def __getitem__(self, index):
        try:
            values = self.data[index]
        except:
            print('Woah')
        target = self.targets[index]
        if self.x_transform is not None:
            values = self.x_transform(values)
        if self.y_transform is not None:
            target = self.y_transform(target)
        return values, target

    def __len__(self):
        return len(self.targets)

