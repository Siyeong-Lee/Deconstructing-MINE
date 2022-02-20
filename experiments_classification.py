import argparse
import torch
import torchvision
import numpy as np
import torchvision.transforms as transforms
from functools import partial
import os
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


def get_accuracy(net, loader, device):
    count = 0
    correct = 0

    with torch.no_grad():
        for i, data in tqdm.tqdm(enumerate(loader, 0)):
            inputs, labels = data

            inputs = inputs.to(device)
            outputs = net(inputs).argmax(dim=1).cpu().detach().numpy()
            labels = labels.numpy()

            count += len(labels)
            correct += sum(labels == outputs)

    return correct / count


def _mask(labels, num_classes):
    return torch.nn.functional.one_hot(labels, num_classes=num_classes).bool()


def _regularized_loss(mi, reg):
    loss = mi - reg
    with torch.no_grad():
        mi_loss = mi - loss
    return loss + mi_loss


def _mine(logits, labels):
    batch_size, classes = logits.shape
    joints = torch.masked_select(logits, _mask(labels, classes))
    t = torch.mean(joints)
    et = torch.logsumexp(logits, dim=(0, 1)) - np.log(batch_size * classes)
    return t, et, joints, logits.flatten()


def _infonce(logits, labels):
    _, classes = logits.shape
    joints = torch.masked_select(logits, _mask(labels, classes))
    t = joints.mean()
    et = logits.logsumexp(dim=1).mean() - np.log(classes)
    return t, et, joints, logits.flatten()


def _smile(logits, labels, clip):
    t, et, _, _ = _mine(torch.clamp(logits, -clip, clip), labels)
    _, _, joints, marginals = _mine(logits, labels)
    return t, et, joints, marginals


def _tuba(logits, labels, clip, a_y):
    _, classes = logits.shape
    joint = torch.masked_select(logits, _mask(labels, classes))
    marginal = logits.flatten()
    if clip > 0.0:
        t = torch.clip(joint, -clip, clip).mean()
        et = torch.clip(marginal, -clip, clip).exp().mean() / a_y + np.log(a_y) - 1.0
    else:
        t = joint.mean()
        et = marginal.exp().mean() / a_y + np.log(a_y) - 1.0

    return t, et, joint, marginal


def _js(logits, labels):
    _, classes = logits.shape
    joints = torch.masked_select(logits, _mask(labels, classes))
    marginals = logits.flatten()
    t = -torch.nn.functional.softplus(-joints).mean()
    et = torch.nn.functional.softplus(marginals).mean()
    return t, et, joints, marginals


def mine(logits, labels):
    t, et, joints, marginals = _mine(logits, labels)
    return t - et, joints, marginals


def mine(logits, labels):
    t, et, joint, marginal = _mine(logits, labels)
    return t - et, joint, marginal


def remine(logits, labels, alpha, bias):
    t, et, joints, marginals = _mine(logits, labels)
    reg = alpha * torch.square(et - bias)
    return  _regularized_loss(t - et, reg), joints, marginals


def infonce(logits, labels):
    t, et, joint, marginal = _infonce(logits, labels)
    return t - et, joint, marginal


def reinfonce(logits, labels, alpha, bias):
    t, et, joint, marginal = _infonce(logits, labels)
    reg = alpha * torch.square(et - bias)
    return _regularized_loss(t - et, reg), joint, marginal


def smile(logits, labels, clip):
    t, et, joint, marginal = _smile(logits, labels, clip)
    return t - et, joint, marginal


def resmile(logits, labels, clip, alpha, bias):
    t, et, joint, marginal = _smile(logits, labels, clip)
    _, reg_et, _, _ = _mine(logits, labels)
    reg = alpha * torch.square(reg_et - bias)
    return _regularized_loss(t - et, reg), joint, marginal


def tuba(logits, labels):
    t, et, joint, marginal = _tuba(logits, labels, 0.0, 1.0)
    return t - et, joint, marginal


def nwj(logits, labels):
    t, et, joint, marginal = _tuba(logits, labels, 0.0, np.e)
    return t - et, joint, marginal


def retuba(logits, labels, clip, alpha):
    t, et, joint, marginal = _tuba(logits, labels, clip, 1.0)
    _, _, _, reg_marginal = _tuba(logits, labels, 0.0, 1.0)
    reg = alpha * torch.square(
        torch.logsumexp(reg_marginal, dim=0) - np.log(reg_marginal.shape[0])
    )
    return _regularized_loss(t - et, reg), joint, marginal


def renwj(logits, labels, clip, alpha):
    t, et, joint, marginal = _tuba(logits, labels, clip, np.e)
    _, _, _, reg_marginal = _tuba(logits, labels, 0.0, np.e)
    reg = alpha * torch.square(
        torch.logsumexp(reg_marginal, dim=0) - np.log(reg_marginal.shape[0])
    )
    return _regularized_loss(t - et, reg), joint, marginal


def js(logits, labels):
    t, et, joint, marginal = _js(logits, labels)
    return t - et, joint, marginal


def rejs(logits, labels, alpha, bias):
    t, et, joint, marginal = _js(logits, labels)
    reg = alpha * torch.square(et - bias)
    return _regularized_loss(t - et, reg), joint, marginal


def nwjjs(logits, labels):
    loss, joint, marginal = js(logits, labels)
    mi, _, _ = nwj(logits, labels)
    with torch.no_grad():
        mi_loss = mi - loss
    return loss + mi_loss, joint, marginal


def renwjjs(logits, labels, alpha, bias, clip):
    loss, joint, marginal = rejs(logits, labels, alpha, bias)
    mi, _, _ = nwj(logits, labels)
    with torch.no_grad():
        mi_loss = mi - loss
    return loss + mi_loss, joint, marginal


criterions = {
    'mine': mine,
    'infonce': infonce,
    'smile_t1': partial(smile, clip=1.0),
    'smile_t10': partial(smile, clip=10.0),
    'tuba': tuba,
    'nwj': nwj,
    'js': js,
    'nwjjs': nwjjs,
}


for alpha in (0.1, 0.01, 0.001):
    criterions[f'remine_a{alpha}_b0'] = partial(remine, alpha=alpha, bias=0)
    criterions[f'reinfonce_a{alpha}_b0'] = partial(reinfonce, alpha=alpha, bias=0)
    criterions[f'resmile_t10_a{alpha}_b0'] = partial(resmile, clip=10, alpha=alpha, bias=0)
    criterions[f'renwj_t10_a{alpha}'] = partial(renwj, clip=10, alpha=alpha)
    criterions[f'retuba_t10_a{alpha}'] = partial(retuba, clip=10, alpha=alpha)
    criterions[f'rejs_a{alpha}_b1'] = partial(rejs, alpha=alpha, bias=1.0)
    criterions[f'renwjjs_a{alpha}_b1'] = partial(renwjjs, alpha=alpha, bias=1.0, clip=0.0)


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
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--loss', type=str, choices=criterions.keys())
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--model', type=str, default='resnet18')

    args = parser.parse_args()
    print(args)

    root_dir = Path(f'cls_exp_re/model={args.model}/dataset={args.dataset}/seed={args.seed}')
    root_dir.mkdir(parents=True, exist_ok=True)

    pretrained_path = root_dir / f'{args.loss}.pth'
    loss_saver = NumpyHistorySaver(float, Path(root_dir), f'{args.loss}_loss')
    accuracy_saver = NumpyHistorySaver(float, Path(root_dir), f'{args.loss}_accuracy')
    joint_saver = NumpyHistorySaver(np.ndarray, Path(root_dir), f'{args.loss}_joint')
    marginal_saver = NumpyHistorySaver(np.ndarray, Path(root_dir), f'{args.loss}_marginal')

    if os.path.exists(pretrained_path):
        print(f'{root_dir}: Results already exists')
        sys.exit(0)

    set_seed(args.seed)

    trainloader, testloader, num_classes = prepare_dataset(args)
    net = getattr(torchvision.models, args.model)(pretrained=False, num_classes=num_classes)

    criterion = criterions[args.loss]
    optimizer = torch.optim.Adam(net.parameters())

    net.to(args.device)

    for epoch in range(args.epochs):  # loop over the dataset multiple times
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
            train_loop.set_description(f'Epoch [{epoch}/{args.epochs}] {loss.item():.4f}')
            loss_saver.store(loss.item())
            joint_saver.store(joints.detach().cpu().numpy())
            marginal_saver.store(marginals.detach().cpu().numpy())

        joint_saver.dump()
        marginal_saver.dump()
        loss_saver.dump()

        accuracy_saver.store(get_accuracy(net, testloader, args.device))
        accuracy_saver.dump()

    print('Finished Training')
    torch.save({
        'epoch': epoch,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, pretrained_path)
