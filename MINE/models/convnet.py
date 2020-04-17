import torch
import torch.nn as nn
import torchvision.models
import pretrainedmodels as ptn
from efficientnet_pytorch import EfficientNet

from .resnet import resnet18


def get_model(key, num_classes, pretrained):
    if pretrained:
        assert num_classes == 1000
        if key == 'efficientnet-b0':
            return EfficientNet.from_pretrained('efficientnet-b0')
        if key in ('mobilenet_v2', 'resnext50_32x4d', 'mnasnet1_0'):
            return getattr(torchvision.models, key)(num_classes=1000, pretrained='imagenet')
        else:
            return getattr(ptn.models, key)(num_classes=1000, pretrained='imagenet')
    else:
        if key == 'efficientnet-b0':
            return EfficientNet.from_name('efficientnet-b0')
        else:
            return getattr(torchvision.models, key)(num_classes=num_classes, pretrained=False)


def split_model(model, finder):
    prev_model, next_model = finder.split(0, model)
    return nn.Sequential(*prev_model), nn.Sequential(*next_model)


def insert_layer(model, layer, index):
    children = list(model.children())
    children.insert(index, layer)
    return nn.Sequential(*children)


def delete_layer(model, index):
    children = list(model.children())
    children.pop(index)
    return nn.Sequential(*children)


class LayerFinder:
    def __init__(self, index, name):
        self.index = index
        self.name = name

    def check(self, index, layer):
        return index == self.index and self.name in str(type(layer))

    def split(self, index, layer):
        assert self.check(index, layer)
        return [layer], []


class ListFinder:
    def __init__(self, index, name, finder):
        self.index = index
        self.name = name
        self.finder = finder

    def check(self, index, layer):
        return index == self.index and self.name in str(type(layer))

    def split(self, index, layer):
        assert self.check(index, layer)
        for child_index, child_layer in enumerate(layer.children()):
            if self.finder.check(child_index, child_layer):
                layers = list(layer.children())
                prev_layers, next_layers = self.finder.split(child_index, child_layer)
                return layers[:child_index] + prev_layers, next_layers + layers[child_index+1:]
        assert False


targets = {
    'resnet18': [
        ListFinder(0, 'ResNet', LayerFinder(3, 'MaxPool2d'))
    ] + [
        ListFinder(0, 'ResNet', LayerFinder(i, 'Sequential')) for i in (4, 5, 6, 7)
    ]
}


class L2Normalize(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs

    def forward(self, x):
        return nn.functional.normalize(x, *self.args, **self.kwargs)


class OneHot(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs

    def forward(self, x):
        return nn.functional.one_hot(x, *self.args, **self.kwargs).float()


class ConvConcatNet(nn.Module):
    def __init__(self, embedding_x, embedding_y, input_size, hidden_size, pretrained_weights):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1, bias=False),
        )
        if pretrained_weights:
            self.network.load_state_dict(pretrained_weights)
        self.embedding_x = embedding_x
        self.embedding_y = embedding_y

    def forward(self, x, y):
        x = self.embedding_x(x)
        y = self.embedding_y(y)
        xy = torch.cat((x, y), dim=1)
        return self.network(xy)

    @classmethod
    def resnet18(cls, index_x, index_y, num_classes, residual_connection, concat_hidden_size, cnn_pretrained_weights, head_pretrained_weights):
        is_featurewise_comparison = (index_x not in ('output', 'label') and index_y not in ('output', 'label'))
        concat_input_size = 512 if is_featurewise_comparison else num_classes

        def _weight_init(m):
            try:
                m.reset_parameters()
            except AttributeError:
                pass

        def _modify_last_layer(m):
            if is_featurewise_comparison:
                m = delete_layer(m, -1)
                m = insert_layer(
                    m, nn.Sequential(nn.Flatten(), L2Normalize(p=2, dim=1)), len(list(m.children()))
                )
            else:
                m = insert_layer(m, nn.Flatten(), -1)
                m = insert_layer(m, nn.Softmax(dim=1), len(list(m.children())))

            return m

        def _split_resnet18(index_):
            model = resnet18(num_classes=num_classes, residual_connection=residual_connection)
            if cnn_pretrained_weights:
                model.load_state_dict(cnn_pretrained_weights)

            if index_ == 'input':
                model = _modify_last_layer(model)
                encoder, decoder = nn.Sequential(nn.Identity()), model
            elif index_ == 'output':
                model = _modify_last_layer(model)
                encoder, decoder = model, nn.Sequential(nn.Identity())
            elif index_ == 'label':
                encoder, decoder = nn.Sequential(OneHot(num_classes=num_classes)), nn.Sequential(nn.Identity())
            else:
                encoder, decoder = split_model(model, targets['resnet18'][index_])
                decoder = _modify_last_layer(decoder)

            decoder.apply(_weight_init)

            return encoder, decoder

        encoder_x, decoder_x = _split_resnet18(index_x)
        encoder_y, decoder_y = _split_resnet18(index_y)

        return cls(decoder_x, decoder_y, concat_input_size, concat_hidden_size, head_pretrained_weights), encoder_x, encoder_y
