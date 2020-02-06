import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class random_number(Dataset):
    def __init__(self, num_classes, length, isOnehot=True):
        self.num_classes = num_classes
        self.isOnehot = isOnehot
        self.length = length

        self.__generate_num__()

    def __generate_num__(self):
        self.x_joint_sample = torch.randint(low=0, 
            high=self.num_classes,
            size=(self.length, ))
        self.y_joint_sample = self.x_joint_sample.clone()

        self.x_marginal_sample = torch.randint(low=0, 
            high=self.num_classes,
            size=(self.length, ))
        self.y_marginal_sample = torch.randint(low=0, 
            high=self.num_classes,
            size=(self.length, ))

        if self.isOnehot:
            self.x_joint_sample = nn.functional.one_hot(
                self.x_joint_sample,
                num_classes=self.num_classes).float()
            self.y_joint_sample = nn.functional.one_hot(
                self.y_joint_sample,
                num_classes=self.num_classes).float()
            self.x_marginal_sample = nn.functional.one_hot(
                self.x_marginal_sample,
                num_classes=self.num_classes).float()
            self.y_marginal_sample = nn.functional.one_hot(
                self.y_marginal_sample,
                num_classes=self.num_classes).float()
     
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        xj_sample = self.x_joint_sample[index]
        yj_sample = self.y_joint_sample[index]

        xm_sample = self.x_marginal_sample[index]
        ym_sample = self.y_marginal_sample[index]
        return (xj_sample, yj_sample), (xm_sample, ym_sample)

def get_data_loaders(batch_size, num_classes, length, isOnehot=True, workers=1):
    # sample 불균일하게 뽑아보는 실험을 위해서 joint marginal 의 배치를 고려해서 설정하는 것이 좋을 것 같음
    raise NotImplementedError
