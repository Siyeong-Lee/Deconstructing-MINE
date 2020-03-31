import torch
import numpy as np
import pandas as pd


class MINEController:
    def __init__(self, data_loader, loss, network, optimizer):
        self.data_loader = iter(data_loader)
        self.network = network
        self.loss = loss
        self.optimizer = optimizer

        self.history = pd.DataFrame()
        self.device = torch.device('cpu')
        self.is_train = True

    def to(self, device):
        self.network.to(device)
        self.device = device

    def clear(self):
        del self.network

    def train(self):
        self.network.train()
        self.is_train = True

    def freeze(self):
        self.network.eval()
        self.is_train = False

    def step(self):
        data = next(self.data_loader)
        for key in data.keys():
            data[key] = data[key].to(device=self.device)

        joint_case = self.network(data['x_joint_sample'], data['y_joint_sample'])
        marginal_case = self.network(data['x_marginal_sample'], data['y_marginal_sample'])

        if self.is_train:
            self.optimizer.zero_grad()
            self.loss(joint_case, marginal_case).backward()
            self.optimizer.step()

        batch_history = data
        batch_history['joint_value'] = joint_case
        batch_history['marginal_value'] = marginal_case
        batch_history = {k: v.detach().cpu().numpy().flatten() for k, v in batch_history.items()}
        batch_history = {k: v for k, v in batch_history.items() if v.shape == batch_history['joint_value'].shape}
        
        self.history = self.history.append(pd.DataFrame(batch_history), ignore_index=True)

    def estimate(self, number_of_samples):
        joint = np.mean(self.history['joint_value'][-number_of_samples:])
        marginal = np.log(np.mean(np.exp(self.history['marginal_value'][-number_of_samples:])))
        return joint - marginal
