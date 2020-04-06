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
    
    def save(self, path):
        torch.save({
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'loss': self.loss,
            'history': self.history,
        }, path)

    def load(self, path):
        ckpt = torch.load(path)
        self.network.load_state_dict(ckpt['network'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.loss = ckpt['loss']
        self.history = ckpt['history']


class ConvFeatureController:
    def __init__(self, dataset, batch_size, x_encoding, y_encoding, label_as_input=False):
        super().__init__()

        self.joint_imageloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.marginal_x_imageloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.marginal_y_imageloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.batch_size = batch_size
        self.iteration = len(self.joint_imageloader)
        self.label_as_input = label_as_input

        self.x_encoding = x_encoding
        self.y_encoding = y_encoding

        self.device = torch.device('cpu')

    def to(self, device):
        self.x_encoding.to(device)
        self.y_encoding.to(device)
        self.device = device

    def __iter__(self):
        while True:
            joint_image = iter(self.joint_imageloader)
            marginal_x_image = iter(self.marginal_x_imageloader)
            marginal_y_image = iter(self.marginal_y_imageloader)

            for _ in range(self.iteration):
                raw_joint_sample, raw_joint_label = next(joint_image)
                raw_x_marginal_sample, raw_x_marginal_label = next(marginal_x_image)
                raw_y_marginal_sample, raw_y_marginal_label = next(marginal_y_image)

                raw_joint_sample = raw_joint_sample.to(self.device)
                raw_x_marginal_sample = raw_x_marginal_sample.to(self.device)

                x_joint_sample = self.x_encoding(raw_joint_sample)
                x_marginal_sample = self.x_encoding(raw_x_marginal_sample)

                if self.label_as_input:
                    raw_joint_label = raw_joint_label.to(self.device)
                    raw_y_marginal_label = raw_y_marginal_label.to(self.device)

                    y_joint_sample = self.y_encoding(raw_joint_label)
                    y_marginal_sample = self.y_encoding(raw_y_marginal_label)
                else:
                    raw_y_marginal_sample = raw_y_marginal_sample.to(self.device)

                    y_joint_sample = self.y_encoding(raw_joint_sample)
                    y_marginal_sample = self.y_encoding(raw_y_marginal_sample)

                sample = {
                    'x_joint_sample': x_joint_sample,
                    'y_joint_sample': y_joint_sample,
                    'x_marginal_sample': x_marginal_sample,
                    'y_marginal_sample': y_marginal_sample,
                    'raw_joint_label': raw_joint_label,
                    'raw_x_marginal_label': raw_x_marginal_label,
                    'raw_y_marginal_label': raw_y_marginal_label,
                }

                yield sample
