import argparse
import functools

import pandas as pd
import pathlib
import tqdm

import torch
import torchvision.transforms
import torch.optim as optim

from MINE import controller, losses, models


parser = argparse.ArgumentParser()

parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--pretrained_model_path', type=str)
parser.add_argument('--model', type=str, choices=('resnet18', ))
parser.add_argument('--compare_to', type=str, choices=('input', 'label', ))
parser.add_argument('--target_index', type=int)
parser.add_argument('--iterations', type=int, default=50000)
parser.add_argument('--batch_size', type=int, default=1000)
parser.add_argument('--concat_hidden_size', type=int, default=1024)

args = parser.parse_args()
print(args)

cifar10 = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
)

if args.model == 'resnet18':
    network, x_encoding, y_encoding = models.ConvConcatNet.resnet18(
        index_x=args.target_index,
        index_y=args.compare_to,
        num_classes=10,
        concat_hidden_size=args.concat_hidden_size,
        cnn_pretrained_weights=torch.load(args.pretrained_model_path),
        head_pretrained_weights=None
    )
else:
    raise NotImplementedError()

data_loader = controller.ConvFeatureController(
    dataset=cifar10,
    batch_size=args.batch_size,
    x_encoding=x_encoding,
    y_encoding=y_encoding,
    label_as_input=(args.compare_to == 'label'),
)

loss = functools.partial(losses.mine_sh_loss, target_value=0, distance_measure='l2')
optimizer = optim.SGD(network.parameters(), lr=0.01)

agent = controller.MINEController(
    data_loader=data_loader, loss=loss, network=network, optimizer=optimizer
)

data_loader.to(args.gpu_id)
agent.to(args.gpu_id)

with tqdm.tqdm(range(args.iterations)) as t:
    for i in t:
        agent.step()
        if i % 10 == 9:
            description = 'MI: %.3f' % agent.estimate(20000)
            t.set_description(description)
            t.refresh()

            folder_name = pathlib.Path(f'history/{args.model}/{args.pretrained_model_path.split("/")[-1]}/{args.compare_to}/{args.target_index}')
            folder_name.mkdir(parents=True, exist_ok=True)
            agent.history.to_pickle(str(folder_name / ('%05d.pkl' % i)))
            agent.history = pd.DataFrame()

            if 'nan' in description or 'inf' in description:
                break
