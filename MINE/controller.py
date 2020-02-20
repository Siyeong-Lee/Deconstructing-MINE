import torch
import numpy as np
import pandas as pd


class MINEController:
    def __init__(self, data_loader, loss, network, optimizer):
        self.data_loader = iter(data_loader)
        self.network = network
        self.loss = loss
        self.optimizer = optimizer

        self.history = pd.DataFrame(columns=('joint_value', 'marginal_value', 'is_joint_case'))
        self.device = torch.device('cpu')

    def to(self, device):
        self.network.to(device)
        self.device = device

    def clear(self):
        del self.network

    def train(self):
        self.network.train()

    def freeze(self):
        self.network.eval()

    def step(self):
        data = next(self.data_loader)
        for key in data.keys():
            data[key] = data[key].to(device=self.device)

        joint_case = self.network(data['x_joint_sample'], data['y_joint_sample'])
        marginal_case = self.network(data['x_marginal_sample'], data['y_marginal_sample'])

        self.optimizer.zero_grad()
        self.loss(joint_case, marginal_case).backward()
        self.optimizer.step()

        self.history = self.history.append(pd.DataFrame({
            'joint_value': joint_case.detach().cpu().numpy().flatten(),
            'marginal_value': marginal_case.detach().cpu().numpy().flatten(),
            'is_joint_case': data['is_joint_case'].detach().cpu().numpy().flatten(),
        }), ignore_index=True)

    def estimate(self, number_of_samples):
        joint = np.mean(self.history['joint_value'][-number_of_samples:])
        marginal = np.log(np.mean(np.exp(self.history['marginal_value'][-number_of_samples:])))
        return joint - marginal
