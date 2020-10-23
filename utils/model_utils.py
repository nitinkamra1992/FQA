import torch
import torch.nn as nn
import torch.nn.functional as F


######## Helper methods ########


def count_parameters(net):
    """ Returns total number of trainable parameters in net """
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


# Reshape layer
class Reshape(nn.Module):
    def __init__(self, out_shape):
        super(Reshape, self).__init__()
        self.out_shape = out_shape

    def forward(self, x):
        return x.view((x.size(0),) + self.out_shape)

    def extra_repr(self):
        return 'out_shape={}'.format(self.out_shape)


# Flatten layer
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


######## Torch aliases ########


optim_list = {
    'sgd': torch.optim.SGD,
    'adagrad': torch.optim.Adagrad,
    'adadelta': torch.optim.Adadelta,
    'adam': torch.optim.Adam,
    'adamax': torch.optim.Adamax,
    'rmsprop': torch.optim.RMSprop,
}


activation_list = {
    'relu': F.relu,
    'sigmoid': F.sigmoid,
    'tanh': torch.tanh,
    'leaky_relu': F.leaky_relu,
    'log_softmax': F.log_softmax,
    'softmax': F.softmax,
    'elu': F.elu,
    'linear': (lambda x: x)
}


layer_list = {
    'Linear': nn.Linear,
    'ReLU': nn.ReLU,
    'Conv2d': nn.Conv2d,
    'Reshape': Reshape,
}