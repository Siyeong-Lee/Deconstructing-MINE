import math

import torch


def _log_mean_exp(tensor):
    batch_size = tensor.shape[0]
    return torch.add(torch.logsumexp(tensor, dim=0), -math.log(batch_size))


def _mean(tensor):
    return torch.mean(tensor)


def _exp(tensor):
    return torch.exp(tensor)


def _l2_distance(tensor, value):
    return torch.pow(tensor - value, 2)


def _l1_distance(tensor, value):
    return torch.abs(tensor - value)


class _base_loss:
    def __init__(self):
        self.t = None
        self.et = None
        self.mi = None
        self.value = None


class smile_loss(_base_loss):
    def __init__(self, clamp_val=5):
        self.clamp_val = clamp_val
        
    def __call__(self, joint, marginal):
        t = _mean(joint)
        et = _log_mean_exp(torch.clamp(marginal, -self.clamp_val, self.clamp_val))
        mi = t - et
        loss = -mi

        self.t, self.et, self.mi, self.value = t.item(), et.item(), mi.item(), loss.item()

        return -mi

    
class mine_loss(_base_loss):
    def __call__(self, joint, marginal):
        t = _mean(joint)
        et = _log_mean_exp(marginal)
        mi = t - et
        loss = -mi

        self.t, self.et, self.mi, self.value = t.item(), et.item(), mi.item(), loss.item()

        return -mi


class nwj_loss(_base_loss):
    def __call__(self, joint, marginal):
        t = _mean(joint)
        et = _mean(_exp(marginal - 1.0))
        mi = t - et
        loss = -mi
        
        self.t, self.et, self.mi, self.value = t.item(), et.item(), mi.item(), loss.item()

        return -mi


class imine_loss(_base_loss):
    def __init__(self, target_value=0, distance_measure='l2', regularizer_weight=1.0):
        self.target_value = target_value
        self.regularizer_weight = regularizer_weight
        if distance_measure == 'l2':
            self.distance_function = _l2_distance
        elif distance_measure == 'l1':
            self.distance_function = _l1_distance
        else:
            raise NotImplementedError(f'No such distance_measure as {distance_measure}')
        
        self.t = None
        self.et = None
        self.mi = None
        self.value = None

    def __call__(self, joint, marginal):
        t = _mean(joint)
        et = _log_mean_exp(marginal)
        mi = t - et
        loss = -mi + self.distance_function(et, self.target_value) * self.regularizer_weight
        
        self.t, self.et, self.mi, self.value = t.item(), et.item(), mi.item(), loss.item()

        return loss


class tuba_loss(_base_loss):
    def __init__(self, log_baseline=None):
        self.log_baseline = log_baseline
   
    def __call__(self, joint, marginal):
        if self.log_baseline is not None:
            joint -= self.log_baseline
            marginal -= self.log_baseline

        t = _mean(joint)
        et = _mean(_exp(marginal))
        mi = 1.0 + t - et
        loss = -mi

        self.t, self.et, self.mi, self.value = t.item(), et.item(), mi.item(), loss.item()
        
        return -mi

class classification_loss(_base_loss):
    def __init__(self):
        self.eps = 1e-8

    def __call__(self, joint, marginal):        
        pos_label_pred_p = F.sigmoid(joint)        
        rn_est_p = (pos_label_pred_p + self.eps / 1-pos_label_pred_p-self.eps) 
        finp_p = torch.log(torch.abs(rn_est_p))

        pos_label_pred_q = F.sigmoid(marginal)
        rn_est_q = (pos_label_pred_q + self.eps / 1-pos_label_pred_q-self.eps) 
        finp_q = torch.log(torch.abs(rn_est_q))

        t = _mean(finp_p) 
        et = _mean(_exp(finp_q))
        mi = t - et
   
        ones  = torch.ones_like(joint).to(joint.device)
        zeros = torch.zeros_like(marginal).to(marginal.device)        
        loss = F.cross_entropy(joint, ones) + F.corss_etropy(marginal, zeros)

        self.t, self.et, self.mi, self.value = t.item(), et.item(), mi.item(), loss.item()
        
        return loss