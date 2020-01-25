import torch
import torch.nn as nn
import torchvision

class random_number(Dataset):
    def __init__(self, batch_size, num_classes, length, isOnehot=True):
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.isOnehot = isOnehot
        self.length = length

    def __len__(self):
        return self.length

    def __getitem(self, index):
        x_joint_sample = torch.randint(low=0, high=self.num_classes,
            size=(self.batch_size, ))
        y_joint_sample = x_joint_sample.clone()

        x_marginal_sample = torch.randint(low=0, high=self.num_classes,
            size=(self.batch_size, ))
        y_marginal_sample = torch.randint(low=0, high=self.num_classes,
            size=(self.batch_size, ))

        if self.isOnehot:
            x_joint_sample = nn.functional.onehot(x_joint_sample,
                num_classes=self.num_classes).float()
            y_joint_sample = nn.functional.onehot(y_joint_sample,
                num_classes=self.num_classes).float()
            x_marginal_sample = nn.functional.onehot(x_marginal_sample,
                num_classes=self.num_classes).float()
            y_marginal_sample = nn.functional.onehot(y_marginal_sample,
                num_classes=self.num_classes).float()

        return (x_joint_sample, y_joint_sample), (x_marginal_sample, y_marginal_sample)

def get_data_loaders(batch_size, num_classes, length, isOnehot=True):
    return random_number(batch_size, num_classes, batch_size*length, isOnehot)
