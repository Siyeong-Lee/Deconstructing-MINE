import math
import torch

def estimate_MI(joint_value, marginal_value, batch_size):
    t = torch.mean(joint_value)
    et = torch.add(torch.logsumexp(marginal_value, dim=0), 
        -math.log(batch_size))

    mi = t - et
    return mi.detach().item()
