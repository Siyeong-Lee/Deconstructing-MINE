import torch
import numpy as np
import pandas as pd


class MINEController:
    def __init__(self, data_loader, loss, network, optimizer, save_input=True, clip_grad=False):
        self.data_loader = iter(data_loader)
        self.network = network
        self.loss = loss
        self.optimizer = optimizer

        self.history = pd.DataFrame()
        self.device = torch.device('cpu')
        self.is_train = True
        self.save_input = save_input
        self.clip_grad = clip_grad

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
            if self.clip_grad:
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.25)
            self.optimizer.step()
        else:
            self.loss(joint_case, marginal_case)

        if self.save_input:
            batch_history = data
        else:
            batch_history = {}
        batch_history['joint_value'] = joint_case
        batch_history['marginal_value'] = marginal_case
        batch_history = {k: v.detach().cpu().numpy() for k, v in batch_history.items()}
        batch_history = {k: v.flatten() if v[0].shape == (1, ) else list(v)  for k, v in batch_history.items()}
        
        self.history = self.history.append(pd.DataFrame(batch_history), ignore_index=True)

    def estimate(self, number_of_samples):
        joint = np.mean(self.history['joint_value'][-number_of_samples:])
        marginal = np.log(np.mean(np.exp(self.history['marginal_value'][-number_of_samples:])))
        return joint - marginal
    
    def save(self, path):
        torch.save({
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'history': self.history,
        }, path)

    def load(self, path):
        ckpt = torch.load(path)
        self.network.load_state_dict(ckpt['network'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.history = ckpt['history']


