import argparse
import functools
import math
import pathlib
import pickle

import numpy as np
import pandas as pd
import tqdm

import torch.utils.data as data
import torch.optim as optim

from MINE import controller, datasets, losses, models


parser = argparse.ArgumentParser()

parser.add_argument('--gpu_id', type=int)
parser.add_argument('--batch_size', type=int)
parser.add_argument('--regularizer_weight', type=float)
parser.add_argument('--iterations', type=int, default=10000)
parser.add_argument('--tries', type=int, default=10)
parser.add_argument('--task_dimension', type=int, default=16)
parser.add_argument('--hidden_dimension', type=int, default=64)
parser.add_argument('--loss', type=str, default='imine')

args = parser.parse_args()
print(args)

data_loader = data.DataLoader(
    datasets.IntegerPairDataset(
        number_of_cases=args.task_dimension,
        x_encoding='one_hot',
        y_encoding='one_hot',
        transform=datasets.TransformToTensor()
    ),
    batch_size=args.batch_size,
)

if args.loss == 'imine':
    loss = losses.imine_loss(target_value=0, distance_measure='l2', regularizer_weight=args.regularizer_weight)
elif args.loss == 'mine':
    loss = losses.mine_loss()
elif args.loss == 'minef':
    loss = losses.minef_loss()
elif args.loss == 'tuba':
    loss = losses.tuba_loss()
elif args.loss == 'nwj':
    loss = losses.nwj_loss()
else:
    raise Exception(f'No such loss {args.loss}')

for try_index in range(args.tries):
    if args.loss == 'imine':
        folder_name = pathlib.Path(
            f'single_history/{args.regularizer_weight}/{args.batch_size}'
        )
    else:
        folder_name = pathlib.Path(
            f'single_history/{args.loss}/{args.batch_size}'
        )
    folder_name.mkdir(parents=True, exist_ok=True)
    file_name = folder_name / f'{try_index}.pkl'
    
    if file_name.exists():
        continue
    
    network = models.ConcatNet(args.task_dimension, args.hidden_dimension)
    optimizer = optim.SGD(network.parameters(), lr=0.1)

    agent = controller.MINEController(
        data_loader=data_loader, loss=loss, network=network, optimizer=optimizer
    )

    agent.to(args.gpu_id)

    loss_values = []
    agent.train()

    for i in range(args.iterations):
        agent.step()
        loss_values.append({
            'loss': agent.loss.value,
            't': agent.loss.t,
            'et': agent.loss.et,
            'mi': agent.loss.mi,
        })

    results = agent.history, pd.DataFrame(loss_values)

    pickle.dump(results, open(str(file_name), 'wb'))

    del agent
