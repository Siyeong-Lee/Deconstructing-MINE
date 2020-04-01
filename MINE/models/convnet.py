import torch
import torch.nn as nn
import torchvision.models
import pretrainedmodels as ptn
from efficientnet_pytorch import EfficientNet


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
    children = children[:index] + [layer] + children[index:]
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

class ConvConcatNet(nn.Module):
    def __init__(self, embedding_x, embedding_y, input_size, hidden_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1, bias=False),
        )
        self.embedding_x = embedding_x
        self.embedding_y = embedding_y

    def forward(self, x, y):
        x = self.embedding_x(x)
        y = self.embedding_y(y)
        xy = torch.cat((x, y), dim=1)
        return self.network(xy)

    @classmethod
    def resnet18(cls, index_x, index_y, num_classes, concat_hidden_size, pretrained_weights):
        def _split_resnet18(index_):
            model = get_model('resnet18', num_classes, pretrained=False)
            if pretrained_weights is not None:
                model.load_state_dict(pretrained_weights)
            encoder, decoder = split_model(model, targets['resnet18'][index_])
            decoder = insert_layer(decoder, nn.Flatten(), len(list(decoder.children()))-1)
            return encoder, decoder

        encoder_x, decoder_x = _split_resnet18(index_x)
        encoder_y, decoder_y = _split_resnet18(index_y)

        return cls(decoder_x, decoder_y, num_classes, concat_hidden_size), encoder_x, encoder_y
