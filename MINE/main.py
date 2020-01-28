from __future__ import print_function
import os
import random
import math
import numpy as np
import argparse

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.nn.parallel

import model_loader
import dataloader
import losses

from lookahead import Lookahead
from torchcontrib.optim import SWA

def init_params(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            init.kaiming_normal_(m.weight, mode='fan_in')
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal_(m.weight, std=1e-3)
            if m.bias is not None:
                init.constant_(m.bias, 0)

# Training
def train(loader, net, criterion, optimizer, prev_et, use_cuda=True):
    net.train()
    loss = 0
    joint_samples, marginal_samples = loader.next()

    if use_cuda:
        joint_samples = joint_samples.cuda()
        marginal_samples = marginal_samples.cuda()
        
    optimizer.zero_grad()
        
    joint_value = net(joint_samples[0], joint_samples[1])
    marginal_value = net(marginal_samples[0], marginal_samples[1])
        
    loss, mi, t, et  = criterion(joint_value, marginal_value)
    
    if abs(et) < abs(prev_et):  
        loss.backward()
        optimizer.step()
    
    return loss, mi, t, et

def name_save_folder(args):
    save_folder = args.model + '_' + str(args.optimizer) + '_lr=' + str(args.lr)
    save_folder += '_cls=' + str(args.num_classes)
    save_folder += '_str=' + str(args.strategy)
    save_folder += '_opt=' + str(args.optimizer)
    save_folder += '_bs=' + str(args.batch_size)
    save_folder += '_wd=' + str(args.weight_decay)
    save_folder += '_mom=' + str(args.momentum)
    save_folder += '_iteration=' + str(args.iteration)
    save_folder += '_loss=' + str(args.loss_name)
    
    if args.ngpu > 1:
        save_folder += '_ngpu=' + str(args.ngpu)
    if args.idx:
        save_folder += '_idx=' + str(args.idx)

    return save_folder

if __name__ == '__main__':
    # Training options
    parser = argparse.ArgumentParser(description='PyTorch MINE Training')
    parser.add_argument('--num_classes', default=4, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--strategy', default='swa', help='strategy: basic | lookahead | swa')
    parser.add_argument('--optimizer', default='sgd', help='optimizer: sgd | adam ')
    parser.add_argument('--weight_decay', default=0.00, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--iteration', default=50000, type=int, metavar='N', help='number of total iteration to run')
    parser.add_argument('--logs', default='logs',help='path to save logs')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--rand_seed', default=0, type=int, help='seed for random num generator')
    parser.add_argument('--resume_model', default='', help='resume model from checkpoint')
    parser.add_argument('--resume_opt', default='', help='resume optimizer from checkpoint')

    # model parameters
    parser.add_argument('--model', '-m', default='concat')
    parser.add_argument('--loss_name', '-l', default='basic', help='loss functions: basic | MINE_stable')

    # data parameters
    parser.add_argument('--idx', default=0, type=int, help='the index for the repeated experiment')

    args = parser.parse_args()

    print('\nLearning Rate: %f' % args.lr)

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print('Current devices: ' + str(torch.cuda.current_device()))
        print('Device count: ' + str(torch.cuda.device_count()))
    else:
        print('Current devices: CPU only')

    # Set the seed for reproducing the results
    random.seed(args.rand_seed)
    np.random.seed(args.rand_seed)
    torch.manual_seed(args.rand_seed)
    if use_cuda:
        torch.cuda.manual_seed_all(args.rand_seed)
        cudnn.benchmark = True

    lr = args.lr  # current learning rate
    start_iteration = 1  # start from epoch 1 or last checkpoint epoch

    if not os.path.isdir(args.logs):
        os.mkdir(args.logs)

    save_folder = name_save_folder(args)
    if not os.path.exists(os.path.join(args.logs,save_folder)):
        os.makedirs(os.path.join(args.logs, save_folder))

    f = open(os.path.join(args.logs, save_folder,'log.out'), 'a')

    loader = dataloader.get_data_loaders(args.batch_size,
        args.num_classes, args.iteration)

    # True MI
    true_mi = math.log(args.num_classes)

    # Model
    if args.resume_model:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        checkpoint = torch.load(args.resume_model)
        net = model_loader.load(args.model, args.num_classes)
        net.load_state_dict(checkpoint['state_dict'])
        start_iteration = checkpoint['iteration'] + 1
    else:
        net = model_loader.load(args.model, args.num_classes)
        print(net)
        init_params(net)

    if args.ngpu > 1:
        net = torch.nn.DataParallel(net)
    
    if args.loss_name == 'basic':
        criterion = losses.MI_BasicLoss(args.batch_size)
    elif args.loss_name == 'MINE_stable':
        criterion = losses.MI_MALoss(args.batch_size) 
    else:
        assert NotImplementedError

    if use_cuda:
        net.cuda()
        criterion = criterion.cuda()

    # Optimizer
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    else:
        optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)


    # Strategy
    if args.strategy == 'basic':
        optimizer = optimizer        
    elif args.strategy == 'lookahead':
        print('[strategy]: lookahead')
        
        la_steps = 16
        la_alpha = 0.001
        optimizer = Lookahead(optimizer, la_steps=la_steps, la_alpha=la_alpha)
    elif args.strategy == 'swa':
        optimizer = SWA(optimizer, swa_start=1, swa_freq=256, swa_lr=0.001)
    else:
        assert NotImplementedError

    if args.resume_opt:
        checkpoint_opt = torch.load(args.resume_opt)
        optimizer.load_state_dict(checkpoint_opt['optimizer'])

    prev_et = np.inf
    for iteration in range(start_iteration, args.iteration + 1):
        loss, est_mi, t, et = train(loader, net, criterion, optimizer, 
            prev_et, use_cuda)

        prev_et = et
        status = 'e: %d loss: %.5f estimiated_mi: %.5f (%.5f)[%.5f, %.5f]' % (iteration, loss, est_mi, true_mi, t, et)
        print(status)
        f.write(status)

    f.close()
