import math
import torch
import torch.nn as nn

def calculate_EMA(prev_value, curr_value, weight):
    assert weight >= 0 and weight <= 1
    return weight*curr_value + (1-weight)*prev_value

class MI_BasicLoss(nn.Module):
    def __init__(self):
        super(MI_BasicLoss, self).__init__()
        self.mutual_information = 0

    def calcuclate_mi(self, joint_value, marginal_value):
        mean_t = torch.mean(joint_value)
        log_mean_et = torch.log(torch.mean(torch.exp(marginal_value)))

        #log_mean_et = torch.add(torch.logsumexp(marginal_value, dim=0),
        #    -math.log(batch_size) )
        self.mutual_information = mean_t - log_mean_et

        return mean_t, log_mean_et

    def calculate_mva(self, prev_value, curr_value, weight):
        return weight*curr_value + (1-weight)*prev_value

class MINE_Loss(MI_BasicLoss):
    def __init__(self):
        super(MINE_Loss, self).__init__()

    def forward(self, joint_value, marginal_value):
        mean_t, log_mean_et = super(MINE_Loss, self).calcuclate_mi(
            joint_value, marginal_value)

        loss = -mean_t + log_mean_et
        return loss

class MINE_stable(MI_BasicLoss):
    def __init__(self, weight=0.5):
        super(MINE_stable, self).__init__()
        self.prev_log_mean_et = None
        self.weight = weight

    def forward(self, joint_value, marginal_value):
        mean_t, log_mean_et = super(MINE_stable, self).calcuclate_mi(
            joint_value, marginal_value)

        if self.prev_log_mean_et:
            log_mean_et = calculate_EMA(self.prev_log_mean_et,
                log_mean_et, self.weight)
            self.prev_log_mean_et = log_mean_et.detach()

        loss = -mean_t + log_mean_et
        return loss

class MINE_const(MI_BasicLoss):
    def __init__(self, weight=0.5):
        super(MINE_stable, self).__init__()
        self.weight = weight

    def forward(self, joint_value, marginal_value):
        mean_t, log_mean_et = super(MINE_stable, self).calcuclate_mi(
            joint_value, marginal_value)

        loss = -mean_t + log_mean_et + log_mean_et**2
        return loss


if __name__ == '__main__':
    criterion1 = MINE_Loss()
    criterion2 = MINE_stable()

    joint_value = torch.randn((1, 1))
    marginal_value = torch.randn((16, 1))

    print(joint_value, marginal_value)
    print(criterion1(joint_value, marginal_value))
    print(criterion1.mutual_information)
    print(criterion2(joint_value, marginal_value))
