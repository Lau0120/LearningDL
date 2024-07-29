import torch.nn as nn


def init_weights(network):
    class_name = network.__class__.__name__
    if class_name.find("Conv") != -1:
        nn.init.normal_(network.weight.data, 0.0, 0.02)
    elif class_name.find("BatchNorm") != -1:
        nn.init.normal_(network.weight.data, 1.0, 0.02)
        nn.init.constant_(network.bias.data, 0)
