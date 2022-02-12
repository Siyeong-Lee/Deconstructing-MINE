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
from pathlib import Path


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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
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


def _regularized_loss(mi, reg):
    loss = mi - reg
    with torch.no_grad():
        mi_loss = mi - loss
    return loss + mi_loss


def _mine(logits, labels):
    batch_size, _ = logits.shape
    joint, marginal = _masked_dot_products(logits, labels)
    t = joint.mean()
    et = torch.logsumexp(marginal, dim=0) - np.log(batch_size * (batch_size - 1))
    return t, et, joint, marginal


def _infonce(logits, labels):
    batch_size, _ = logits.shape
    marginal_mask = ~torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    joint_mask = (labels[:, None] == labels) & marginal_mask
    dot_mat = torch.matmul(logits, logits.T)

    joints = torch.masked_select(dot_mat, joint_mask)
    t = joints.mean()

    marginals = torch.masked_select(dot_mat, marginal_mask)
    et_all = torch.logsumexp(marginals.reshape((batch_size, batch_size - 1)), dim=1) - np.log(batch_size - 1)
    et_select = joint_mask.sum(1).float()
    et = torch.dot(et_all, et_select) / et_select.sum()

    return t, et, joints, marginals


def infonce(logits, labels):
    t, et, joint, marginal = _infonce(logits, labels)
    return t - et, joint, marginal


def reinfonce(logits, labels, alpha, bias):
    t, et, joint, marginal = _infonce(logits, labels)
    reg = alpha * torch.square(et - bias)
    return _regularized_loss(t - et, reg), joint, marginal


def mine(logits, labels):
    t, et, joint, marginal = _mine(logits, labels)
    return t - et, joint, marginal


def remine_j(logits, labels):
    t, _, joint, marginal = _mine(logits, labels)
    return t, joint, marginal


def remine(logits, labels, alpha, bias):
    t, et, joint, marginal = _mine(logits, labels)
    reg = alpha * torch.square(et - bias)
    return _regularized_loss(t - et, reg), joint, marginal


def _smile(logits, labels, clip):
    batch_size, _ = logits.shape
    joint, marginal = _masked_dot_products(logits, labels)
    t = torch.clamp(joint, -clip, clip).mean()
    et = torch.logsumexp(torch.clamp(marginal, -clip, clip), dim=0) - np.log(batch_size * (batch_size - 1))
    return t, et, joint, marginal


def smile(logits, labels, clip):
    t, et, joint, marginal = _smile(logits, labels, clip)
    return t - et, joint, marginal


def resmile(logits, labels, clip, alpha, bias):
    t, et, joint, marginal = _smile(logits, labels, clip)
    _, reg_et, _, _ = _mine(logits, labels)
    reg = alpha * torch.square(reg_et - bias)
    return _regularized_loss(t - et, reg), joint, marginal


def _tuba(logits, labels, clip):
    joint, marginal = _masked_dot_products(logits, labels)
    if clip > 0.0:
        joint = torch.clip(joint, -clip, clip)
        marginal = torch.clip(marginal, -clip, clip)

    t = joint.mean()
    et = marginal.exp().mean()
    return t, et, joint, marginal


def tuba(logits, labels, clip):
    t, et, joint, marginal = _tuba(logits, labels, clip)
    return 1 + t - et, joint, marginal


def nwj(logits, labels, clip):
    return tuba(logits - 1.0, labels, clip)


def retuba(logits, labels, clip, alpha, bias):
    t, et, joint, marginal = _tuba(logits, labels, clip)
    reg = alpha * torch.nn.functional.smooth_l1_loss(
        et, torch.tensor(bias).float().to(et.device))
    return _regularized_loss(1 + t - et, reg), joint, marginal


def renwj(logits, labels, clip, alpha, bias):
    return retuba(logits - 1.0, labels, clip, alpha, bias)


def _js(logits, labels):
    joint, marginal = _masked_dot_products(logits, labels)
    t = -torch.nn.functional.softplus(-joint).mean()
    et = torch.nn.functional.softplus(marginal).mean()
    return t, et, joint, marginal


def js(logits, labels):
    t, et, joint, marginal = _js(logits, labels)
    return t - et, joint, marginal


def rejs(logits, labels, alpha, bias):
    t, et, joint, marginal = _js(logits, labels)
    reg = alpha * torch.square(et - bias)
    return _regularized_loss(t - et, reg), joint, marginal


def mix(logits, labels, estimator, trainer):
    mi = estimator(logits, labels)
    grad = trainer(logits, labels)
    with torch.no_grad():
        mi_grad = mi - grad
    return grad + mi_grad


criterions = {
    'mine': mine,
    'infonce': infonce,
    'smile_t1': partial(smile, clip=1.0),
    'smile_t10': partial(smile, clip=10.0),
    'tuba': partial(tuba, clip=0.0),
    'nwj': partial(nwj, clip=0.0),
    'js': js,
    'nwjjs': partial(mix, estimator=nwj, trainer=js),
}

for alpha in (0.1, 0.01, 0.001):
    criterions[f'remine_a{alpha}_b0'] = partial(remine, alpha=alpha, bias=0)
    criterions[f'reinfonce_a{alpha}_b0'] = partial(reinfonce, alpha=alpha, bias=0)
    criterions[f'resmile_t10_a{alpha}_b0'] = partial(resmile, clip=10, alpha=alpha, bias=0)
    criterions[f'renwj_t10_a{alpha}_b0.5'] = partial(renwj, clip=10, alpha=alpha, bias=0.5)
    criterions[f'retuba_t10_a{alpha}_b0.5'] = partial(retuba, clip=10, alpha=alpha, bias=0.5)
    criterions[f'rejs_a{alpha}_b0'] = partial(rejs, alpha=alpha, bias=0)
    criterions[f'renwjjs_a{alpha}_b0'] = partial(mix, estimator=nwj, trainer=partial(rejs, alpha=alpha, bias=0)),


class NumpyHistorySaver:
    def __init__(self, store_type: type, directory: Path, filename: str, prev_iter: int = 0):
        assert store_type in (float, np.ndarray)

        self.directory = directory
        self.filename = filename
        self.store_type = store_type
        self.iteration = prev_iter
        self.history = []

    def get_filename(self):
        return self.directory / f'{self.filename}_{self.iteration}.npy'

    def dump(self):
        if self.store_type == float:
            self.history = np.array(self.history)
        elif self.store_type == np.ndarray:
            self.history = np.concatenate(self.history)

        np.save(self.get_filename(), self.history)
        self.iteration += 1
        self.history = []

    def store(self, value):
        assert isinstance(value, self.store_type)
        self.history.append(value)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--dataset', type=str, choices=available_datasets.keys())
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--loss', type=str, choices=criterions.keys())
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--remove_fc', action='store_true')
    parser.add_argument('--optimizer', type=str, choices=['Adam', 'SGD', 'Adagrad'], default='Adam')
    parser.add_argument('--lr', type=float, default=0.001)

    args = parser.parse_args()
    print(args)

    root_dir = Path(f'self_exp_re/model={args.model}/dataset={args.dataset}/remove_fc={args.remove_fc}/optimizer={args.optimizer}/batch_size={args.batch_size}/lr={args.lr}/seed={args.seed}')
    root_dir.mkdir(parents=True, exist_ok=True)

    if (root_dir / f'{args.loss}.pth').exists():
        print(f'{root_dir}: Results already exists')
        sys.exit(0)

    set_seed(args.seed)

    trainloader, testloader, num_classes = prepare_dataset(args)
    net = getattr(torchvision.models, args.model)(pretrained=False, num_classes=num_classes)
    if args.remove_fc:
        net.fc = torch.nn.Identity()
    net.to(args.device)

    criterion = criterions[args.loss]
    optimizer = getattr(torch.optim, args.optimizer)(net.parameters(), lr=args.lr)

    pretrained_path = root_dir / f'{args.loss}.pth'
    loss_saver = NumpyHistorySaver(float, Path(root_dir), f'{args.loss}_loss')
    accuracy_saver = NumpyHistorySaver(float, Path(root_dir), f'{args.loss}_accuracy')
    joint_saver = NumpyHistorySaver(np.ndarray, Path(root_dir), f'{args.loss}_joint')
    marginal_saver = NumpyHistorySaver(np.ndarray, Path(root_dir), f'{args.loss}_marginal')

    if os.path.exists(pretrained_path):
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        last_epoch = checkpoint['epoch']
    else:
        last_epoch = -1

    nan_count = 0
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
            loss, joints, marginals = criterion(outputs, labels)
            (-loss).backward()
            optimizer.step()

            # print statistics
            train_loop.set_description(f'Epoch [{epoch}/{args.epochs}] ({nan_count}) {loss.item():.4f}')
            loss_saver.store(loss.item())
            joint_saver.store(joints.detach().cpu().numpy())
            marginal_saver.store(marginals.detach().cpu().numpy())

            # Counting NaNs. If too much, break
            if torch.isnan(loss):
                nan_count += 1

        joint_saver.dump()
        marginal_saver.dump()
        loss_saver.dump()

        accuracy_saver.store(get_accuracy(net, trainloader, testloader, args.device))
        accuracy_saver.dump()

        if nan_count >= 1000:
            break

    print('Finished Training')
    torch.save({
        'epoch': epoch,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, pretrained_path)
