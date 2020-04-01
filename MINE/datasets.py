import math

import numpy as np
import torch
import torch.utils.data as data


class RandomGeneratorMixin:
    def sample(self):
        raise NotImplementedError

    def conditional(self, x):
        raise NotImplementedError


class RandomIntegerGenerator(RandomGeneratorMixin):
    def __init__(self, max_value, encoding='identity'):
        self.max_value = max_value

        self._encoding_dict = {
            'one_hot': self._one_hot,
            'square': self._square,
            'cube': self._square,
            'sin': self._sin,
            'cos': self._cos,
            'identity': self._identity,
        }
        if encoding not in self._encoding_dict.keys():
            raise NotImplementedError(f'No such encoding: {encoding}')
        self.encoding = self._encoding_dict[encoding]

    def sample(self):
        s = np.array((np.random.randint(self.max_value), )).astype(float)
        return self.encoding(s), s

    def conditional(self, x):
        return self.encoding(x), x

    def _one_hot(self, x):
        array = np.zeros((self.max_value, ))
        array[int(x)] = 1.0
        return array
    
    def _square(self, x):
        return x * x
    
    def _cube(self, x):
        return x * x * x
    
    def _sin(self, x):
        return np.array((math.sin(x / self.max_value), ))
    
    def _cos(self, x):
        return np.array((math.cos(x / self.max_value), ))

    def _identity(self, x):
        return x


class IntegerPairDataset(data.dataset.IterableDataset):
    def __init__(self, number_of_cases, x_encoding='identity', y_encoding='identity', transform=None):
        super().__init__()
        self.x = RandomIntegerGenerator(max_value=number_of_cases, encoding=x_encoding)
        self.y = RandomIntegerGenerator(max_value=number_of_cases, encoding=y_encoding)
        self.transform = transform

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            x_joint_sample, x_joint_sample_original = self.x.sample()
            y_joint_sample, _ = self.y.conditional(x_joint_sample_original)

            x_marginal_sample, x_marginal_sample_original = self.x.sample()
            y_marginal_sample, y_marginal_sample_original = self.y.sample()
            is_joint_case = x_marginal_sample_original == y_marginal_sample_original

            sample = {
                'x_joint_sample': x_joint_sample,
                'y_joint_sample': y_joint_sample,
                'x_marginal_sample': x_marginal_sample,
                'y_marginal_sample': y_marginal_sample,
                'is_joint_case': is_joint_case,
            }

            if self.transform:
                sample = self.transform(sample)

            return sample


class CorrelatedGaussianDataset(data.dataset.IterableDataset):
    def __init__(self, var, rho, transform=None):
        super().__init__()
        self.generator = lambda: np.random.multivariate_normal(
            mean=(0, 0), cov=((var, rho*var), (rho*var, var))
        )
        self.transform = transform

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            x_joint_sample, y_joint_sample = self.generator()
            x_marginal_sample, _ = self.generator()
            _, y_marginal_sample = self.generator()

            sample = {
                'x_joint_sample': np.array([x_joint_sample]),
                'y_joint_sample': np.array([y_joint_sample]),
                'x_marginal_sample': np.array([x_marginal_sample]),
                'y_marginal_sample': np.array([y_marginal_sample]),
            }

            if self.transform:
                sample = self.transform(sample)

            return sample


class TransformToTensor:
    def __call__(self, sample):
        for key in sample.keys():
            sample[key] = torch.from_numpy(sample[key]).float()
        return sample
