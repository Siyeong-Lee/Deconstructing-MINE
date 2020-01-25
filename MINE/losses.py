import math
import torch
import torch.nn as nn

class MI_BasicLoss(nn.Module):
    def __init__(self, batch_size):
        super(MI_BasicLoss, self).__init__()
        self.batch_size = batch_size

    def forward(self, joint_value, marginal_value):
        t = torch.mean(joint_value)
        et = torch.add(torch.logsumexp(marginal_value, dim=0), 
            -math.log(self.batch_size))

        mi = t - et
        loss = -(t - et)
        return loss, mi

class MI_MALoss(nn.Module):
    def __init__(self, batch_size, weight=0.5):
        super(MI_MALoss, self).__init__()
        self.batch_size = batch_size
        self.prev_et = None
        self.weight = weight

    def forward(self, joint_value, marginal_value):
        t = torch.mean(joint_value)
        et = torch.add(torch.logsumexp(marginal_value, dim=0),
            - math.log(self.batch_size))

        mi = t - et
        if self.prev_et:
            et = self.weight*et + (1-self.weight) * self.prev_et
        
        self.prev_et = et.item()
        loss = -(t - et)

        return loss, mi

class MI_SHLoss(nn.Module):
    def __init__(self, batch_size):
        super(MI_MALoss, self).__init__()
        self.batch_size = batch_size

    def forward(self, joint_value, marginal_value):
        t = torch.mean(joint_value)
        et = torch.add(torch.logsumexp(marginal_value, dim=0),
            - math.log(self.batch_size))

        mi = t - et
        loss = -(t - et * et) 
        return loss, mi
