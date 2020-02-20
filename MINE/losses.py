import math

import torch


def _log_mean_exp(tensor):
    batch_size = tensor.shape[0]
    return torch.add(torch.logsumexp(tensor, dim=0), -math.log(batch_size))


def _mean(tensor):
    return torch.mean(tensor)


def _l2_distance(tensor, value):
    return torch.pow(tensor - value, 2)


def _l1_distance(tensor, value):
    return torch.abs(tensor - value)


def mine_loss(joint, marginal):
    t = _mean(joint)
    et = _log_mean_exp(marginal)
    mi = t - et
    return -mi


def mine_sh_loss(joint, marginal, target_value=0, distance_measure='l2'):
    t = _mean(joint)
    et = _log_mean_exp(marginal)
    mi = t - et

    if distance_measure == 'l2':
        d = _l2_distance(et, target_value)
    elif distance_measure == 'l1':
        d = _l1_distance(et, target_value)
    else:
        raise NotImplementedError(f'No such distance_measure as {distance_measure}')

    return -mi + d
