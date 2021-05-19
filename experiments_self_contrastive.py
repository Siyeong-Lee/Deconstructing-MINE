import argparse
import torch
import torchvision
import numpy as np
import torchvision.transforms as transforms
from functools import partial
import os
import json
import tqdm
import random
import sys


class ModifiedImageNet(torchvision.datasets.ImageNet):
    def __init__(self, root, train, download, transform):
        super().__init__(
            root='./data/imagenet',
            split='train' if train else 'val',
            transform=transform,
        )

    def parse_archives(self):
        pass


available_datasets = {
    'cifar10': (torchvision.datasets.CIFAR10, 10),
    'cifar100': (torchvision.datasets.CIFAR100, 100),
    'mnist': (torchvision.datasets.MNIST, 10),
    'fashion_mnist': (torchvision.datasets.FashionMNIST, 10),
    'imagenet': (ModifiedImageNet, 1000),
}


def set_seed(seed):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True


def prepare_dataset(args):
    dataset_class, num_classes = available_datasets[args.dataset]

    transforms_list = [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
    # Grayscale datasets
    if args.dataset in ('mnist', 'fashion_mnist'):
        transforms_list.insert(0, transforms.Grayscale(num_output_channels=3))

    transform = transforms.Compose(transforms_list)

    trainset = dataset_class(
        root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=8)

    testset = dataset_class(
        root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=8)

    return trainloader, testloader, num_classes


def get_logits_labels(model, loader, device):
    outputs_acc = []
    labels_acc = []

    with torch.no_grad():
        for inputs, labels in tqdm.tqdm(loader):
            outputs = model(inputs.to(device))
            outputs_acc.append(outputs.detach().cpu().numpy())
            labels_acc.append(labels.numpy())

    labels_acc = np.concatenate(labels_acc, axis=0)
    outputs_acc = np.concatenate(outputs_acc, axis=0)

    return outputs_acc, labels_acc


def get_accuracy(model, trainloader, testloader, device):
    train_logits, train_labels = get_logits_labels(model, trainloader, device)
    test_logits, test_labels = get_logits_labels(model, testloader, device)

    count = 0
    correct = 0

    for test_index in tqdm.tqdm(range(len(test_labels))):
        correct += train_labels[np.dot(train_logits, test_logits[test_index]).argmax()] == test_labels[test_index]
        count += 1

    return correct / count


def _masked_dot_products(logits, labels):
    marginal_mask = ~torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    joint_mask = (labels[:, None] == labels) & marginal_mask
    dot_mat = torch.matmul(logits, logits.T)
    return torch.masked_select(dot_mat, joint_mask), torch.masked_select(dot_mat, marginal_mask)


def _mine(logits, labels):
    batch_size, _ = logits.shape
    joint, marginal = _masked_dot_products(logits, labels)
    t = joint.mean()
    et = torch.logsumexp(marginal, dim=0) - np.log(batch_size * (batch_size - 1))
    return t, et


def mine(logits, labels):
    t, et = _mine(logits, labels)
    return t - et


def remine_j(logits, labels):
    t, _ = _mine(logits, labels)
    return t


def remine(logits, labels, alpha, bias):
    t, et = _mine(logits, labels)
    return t - et - alpha * torch.square(et - bias)


def smile(logits, labels, clip):
    batch_size, _ = logits.shape
    joint, marginal = _masked_dot_products(logits, labels)
    t = torch.clamp(joint, -clip, clip).mean()
    et = torch.logsumexp(torch.clamp(marginal, -clip, clip), dim=0) - np.log(batch_size * (batch_size - 1))
    return t - et


def remine_l1(logits, labels, alpha):
    t, et = _mine(logits, labels)
    return t - et - alpha * torch.abs(et)


def tuba(logits, labels):
    joint, marginal = _masked_dot_products(logits, labels)
    t = joint.mean()
    et = marginal.exp().mean()
    return 1 + t - et


def nwj(logits, labels):
    return tuba(logits - 1.0, labels)


def js(logits, labels):
    joint, marginal = _masked_dot_products(logits, labels)
    t = -torch.nn.functional.softplus(-joint).mean()
    et = torch.nn.functional.softplus(marginal).mean()
    return t - et


def mix(logits, labels, estimator, trainer):
    mi = estimator(logits, labels)
    grad = trainer(logits, labels)
    with torch.no_grad():
        mi_grad = mi - grad
    return grad + mi_grad


criterions = {
    'mine': mine,
    'remine-0.1': partial(remine, alpha=0.1, bias=0.0),
    'remine-0.5': partial(remine, alpha=0.5, bias=0.0),
    'remine-1.0': partial(remine, alpha=1.0, bias=0.0),
    'remine_log10-10.0': partial(remine, alpha=10.0, bias=-np.log(10)),
    'remine_log10-1.0': partial(remine, alpha=1.0, bias=-np.log(10)),
    'remine_log10-0.1': partial(remine, alpha=0.1, bias=-np.log(10)),
    'remine_log10-0.01': partial(remine, alpha=0.01, bias=-np.log(10)),
    'remine_log10-0.001': partial(remine, alpha=0.001, bias=-np.log(10)),
    'remine_log100-0.1': partial(remine, alpha=0.1, bias=-np.log(100)),
    'remine_log100-0.01': partial(remine, alpha=0.01, bias=-np.log(100)),
    'remine_l1-0.01': partial(remine_l1, alpha=0.01),
    'remine_l1-0.1': partial(remine_l1, alpha=0.1),
    'remine_l1-1.0': partial(remine_l1, alpha=1.0),
    'smile-1.0': partial(smile, clip=1.0),
    'smile-10.0': partial(smile, clip=10.0),
    'tuba': tuba,
    'nwj': nwj,
    'js': js,
    'nwj_js': partial(mix, estimator=nwj, trainer=js),
    'smile-1.0_js': partial(mix,
                            estimator=partial(smile, clip=1.0),
                            trainer=js),
    'smile-10.0_js': partial(mix,
                             estimator=partial(smile, clip=10.0),
                             trainer=js),
    'smile-1.0_remine-0.1': partial(mix,
                                    estimator=partial(smile, clip=1.0),
                                    trainer=partial(remine, alpha=0.1, bias=0.0)),
    'smile-1.0_remine-1.0': partial(mix,
                                    estimator=partial(smile, clip=1.0),
                                    trainer=partial(remine, alpha=1.0, bias=0.0)),
    'smile-10.0_remine-0.1': partial(mix,
                                     estimator=partial(smile, clip=10.0),
                                     trainer=partial(remine, alpha=0.1, bias=0.0)),
    'smile-10.0_remine-1.0': partial(mix,
                                     estimator=partial(smile, clip=10.0),
                                     trainer=partial(remine, alpha=1.0, bias=0.0)),
    'remine_j-0.1': partial(mix,
                            estimator=remine_j,
                            trainer=partial(remine, alpha=0.1, bias=0.0)),
    'remine_j-1.0': partial(mix,
                            estimator=remine_j,
                            trainer=partial(remine, alpha=1.0, bias=0.0)),
    'remine_j_l1-0.1': partial(mix,
                               estimator=remine_j,
                               trainer=partial(remine_l1, alpha=0.1)),
    'remine_j_l1-1.0': partial(mix,
                               estimator=remine_j,
                               trainer=partial(remine_l1, alpha=1.0)),
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--dataset', type=str, choices=available_datasets.keys())
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--loss', type=str, choices=criterions.keys())
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()
    print(args)

    root_dir = f'self_exp/model={args.model}/dataset={args.dataset}/seed={args.seed}'
    os.makedirs(root_dir, exist_ok=True)

    if os.path.exists(f'{root_dir}/{args.loss}.pth'):
        print(f'{root_dir}: Results already exists')
        sys.exit(0)

    set_seed(args.seed)

    trainloader, testloader, num_classes = prepare_dataset(args)
    net = getattr(torchvision.models, args.model)(pretrained=False, num_classes=num_classes)

    criterion = criterions[args.loss]
    optimizer = torch.optim.Adam(net.parameters())

    net.to(args.device)

    pretrained_path = f'{root_dir}/{args.loss}.pth'
    loss_history_path = f'{root_dir}/{args.loss}.npy'
    train_acc_history_path = f'{root_dir}/{args.loss}.json'

    loss_history = []
    test_acc_history = []

    if os.path.exists(pretrained_path):
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        last_epoch = checkpoint['epoch']
        loss_history = np.load(loss_history_path).tolist()
        test_acc_history = json.load(open(train_acc_history_path))['test_acc']
    else:
        last_epoch = -1

    for epoch in range(last_epoch + 1, args.epochs):  # loop over the dataset multiple times
        train_loop = tqdm.tqdm(enumerate(trainloader, 0))
        for i, data in train_loop:
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            inputs = inputs.to(args.device)
            labels = labels.to(args.device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = -criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            train_loop.set_description(f'Epoch [{epoch}/{args.epochs}] {loss.item():.4f}')
            loss_history.append(loss.item())

        test_acc_history.append(get_accuracy(net, trainloader, testloader, args.device))

        np.save(loss_history_path, np.array(loss_history))
        json.dump({
            'test_acc': test_acc_history,
        }, open(train_acc_history_path, 'w'))

    print('Finished Training')
    torch.save({
        'epoch': epoch,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, pretrained_path)
