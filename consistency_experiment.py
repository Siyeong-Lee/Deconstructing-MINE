import tqdm
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.utils.data as data
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from torchvision.datasets import MNIST
from torchvision.datasets import CIFAR10

from MINE import controller, datasets, losses, models
from torchvision import transforms


class SelfConsistencyDataset(data.IterableDataset):
    def __init__(
        self,
        dataset_name,
        consistency_type,
        used_rows_1,
        used_rows_2=None,
        transform=None
    ):
        super().__init__()
        
        assert dataset_name in ('MNIST', 'cifar10')
        if dataset_name == 'MNIST':
            assert 0 <= used_rows_1 < 28
        else:
            assert 0 <= used_rows_1 < 32

        if used_rows_2 is not None:
            assert used_rows_1 >= used_rows_2
            if dataset_name == 'MNIST':
                assert 0 <= used_rows_2 < 28
            else:
                assert 0 <= used_rows_2 < 32
            assert consistency_type == 2
        else:
            assert consistency_type in (1, 3)

        if dataset_name == 'MNIST':
            self.dataset = MNIST('./MINE/general_dataset/MNIST', train=True, download=True)
        elif dataset_name == 'cifar10':
            self.dataset = CIFAR10('./MINE/general_dataset/cifar10', train=True, download=True)
        else:
            raise NotImplementedError

        self.dataset_name = dataset_name
        self.consistency_type = consistency_type
        self.used_rows_1 = used_rows_1
        self.used_rows_2 = used_rows_2
        self.transform = transform
 
    def type1(self, x):
        y = np.copy(x)
        y[:, self.used_rows_1:, :] = 0
        return x, y
    
    def type2(self, x):        
        y1, y2 = np.copy(x), np.copy(x)
        y1[:, self.used_rows_1:, :] = 0
        y2[:, self.used_rows_2:, :] = 0        
        
        x = np.concatenate((x, x), 0)
        y = np.concatenate((y1, y2), 0)
        return x, y
        
    def type3(self, x1, x2):        
        y1, y2 = np.copy(x1), np.copy(x2)
        y1[:, self.used_rows_1:, :] = 0
        y2[:, self.used_rows_1:, :] = 0        
        
        x = np.concatenate((x1, x2), 0)
        y = np.concatenate((y1, y2), 0)        
        return x, y
    
    def get_single(self):
        idx = random.randint(0, len(self.dataset)-1)
        x, _ = self.dataset[idx]
        x = np.array(x)
        if self.dataset_name == 'MNIST':
            return np.expand_dims(x, 0) / 255.0
        else:
            return np.moveaxis(x, -1, 0) / 255.0

    def get_pair(self):
        x = self.get_single()

        if self.consistency_type == 1:
            x, y = self.type1(x)
        elif self.consistency_type == 2:
            x, y = self.type2(x)
        else:
            x1 = self.get_single()
            x, y = self.type3(x, x1)
        return x, y

    def __iter__(self):
        return self

    def __next__(self):
        x_joint_sample, y_joint_sample = self.get_pair()
        x_marginal_sample, _ = self.get_pair()
        _, y_marginal_sample = self.get_pair()
        sample = {
            'x_joint_sample': x_joint_sample,
            'y_joint_sample': y_joint_sample,
            'x_marginal_sample': x_marginal_sample,
            'y_marginal_sample': y_marginal_sample,
        }
        if self.transform:
            sample = self.transform(sample)
        return sample


    
class Net(nn.Module):
    def __init__(self, input_channels, final_size):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)
        self.fc1 = nn.Linear(128*final_size, 1024)
        self.fc2 = nn.Linear(1024, 1)

    def forward(self, x, y):
        x = torch.cat((x, y), dim=1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

def experiment(loss, dataset_name, consistency_type, used_rows_1, used_rows_2=None):
    if consistency_type == 2 or consistency_type == 3:
        input_channels = 4
    else:
        input_channels = 2

    if dataset_name == 'cifar10':
        input_channels *= 3
        final_size = 64
        iterations = 100
    else:
        final_size = 49
        iterations = 120

    data_loader = data.DataLoader(
        SelfConsistencyDataset(
            dataset_name=dataset_name,
            consistency_type=consistency_type,
            used_rows_1=used_rows_1,
            used_rows_2=used_rows_2,
            transform=datasets.TransformToTensor(),
        ),
        batch_size=100,
    )
    if loss == 'imine':
        loss = losses.imine_loss(regularizer_weight=1)
    elif loss == 'mine':
        loss = losses.mine_loss()
    elif loss == 'smile':
        loss = losses.smile_loss(100)
    else:
        raise Exception('no such loss')

    network = Net(input_channels, final_size)
    optimizer = optim.Adam(network.parameters(), lr=0.0001)

    agent = controller.MINEController(
        data_loader=data_loader, loss=loss, network=network, optimizer=optimizer, save_input=False, clip_grad=False
    )
    
    agent.to(0)
    agent.train()
    for i in tqdm.tqdm(range(iterations)):
        agent.step()

    return agent

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--iteration', type=int)
    parser.add_argument('--loss', type=str)
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--consistency_type', type=int)
    parser.add_argument('--used_rows', type=int)
    args = parser.parse_args()

    if args.consistency_type == 2:
        args.used_rows_2 = args.used_rows - 3
    else:
        args.used_rows_2 = None

    fname = f'consistency_results/{args.loss}_{args.dataset_name}_{args.consistency_type}_{args.used_rows}_{args.iteration}.pkl'
    experiment(args.loss, args.dataset_name, args.consistency_type, args.used_rows, args.used_rows_2).history.to_pickle(fname)
