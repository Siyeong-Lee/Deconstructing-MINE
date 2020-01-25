import os
import torch, torchvision
import models.concat as concatnet
import models.embedded as embeddednet

# map between model name and function
models = {
    'concat'   : concatnet.ConcatNet20,
    'embedded' : embeddednet.EmbeddedNet20
}

def load(model_name, num_classes, model_file=None, data_parallel=False):
    net = models[model_name](num_classes)
    if data_parallel: # the model is saved in data paralle mode
        net = torch.nn.DataParallel(net)

    if model_file:
        assert os.path.exists(model_file), model_file + " does not exist."
        stored = torch.load(model_file, map_location=lambda storage, loc: storage)
        if 'state_dict' in stored.keys():
            net.load_state_dict(stored['state_dict'])
        else:
            net.load_state_dict(stored)

    if data_parallel: # convert the model back to the single GPU version
        net = net.module

    return net
