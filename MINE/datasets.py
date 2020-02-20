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
    def __init__(self, max_value, embedding='identity'):
        self.max_value = max_value

        self._embedding_dict = {
            'one_hot': self._one_hot,
            'square': self._square,
            'cube': self._square,
            'sin': self._sin,
            'cos': self._cos,
            'identity': self._identity,
        }
        if embedding not in self._embedding_dict.keys():
            raise NotImplementedError(f'No such embedding: {embedding}')
        self.embedding = self._embedding_dict[embedding]

    def sample(self):
        s = np.array((np.random.randint(self.max_value), )).astype(float)
        return self.embedding(s), s

    def conditional(self, x):
        return self.embedding(x), x

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
    def __init__(self, number_of_cases, x_embedding='identity', y_embedding='identity', transform=None):
        super().__init__()
        self.x = RandomIntegerGenerator(max_value=number_of_cases, embedding=x_embedding)
        self.y = RandomIntegerGenerator(max_value=number_of_cases, embedding=y_embedding)
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


class TransformToTensor:
    def __call__(self, sample):
        for key in sample.keys():
            sample[key] = torch.from_numpy(sample[key]).float()
        return sample
