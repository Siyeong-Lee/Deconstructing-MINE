"""
    The calculation to be performed at each point (modified model), evaluating
    the loss value, accuracy and eigen values of the hessian matrix
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import estimate_MI

def eval_loss(net, criterion, loader, use_cuda=True):
    loss = 0
    mi = 0, prev = 0
    num_batch = len(loader)
    
    if use_cuda:
        net.cuda()
    net.eval()
    
    for batch_idx, (joint_samples, marginal_samples) in enumerate(loader):
        batch_size = joint_samples.size(0)
        if use_cuda:
            joint_samples = joint_samples.cuda()
            marginal_samples = marginal_samples.cuda()
        
        joint_value = net(joint_samples[0], joint_samples[1])
        marginal_value = net(marginal_samples[0], marginal_samples[1])
        
        loss += criterion(joint_value, marginal_value)

        mi = estimate_MI(joint_value, marginal_value, batch_size)
        mi = 0.1 * prev + 0.9 *mi

    return loss / num_batch 


