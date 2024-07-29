import torch.nn as nn

from util import init_weights
from config import device


class GeneratorBlock(nn.Module):
    def __init__(self, ic, oc, ks, st, pd, bcnm=True, relu=True):
        super(GeneratorBlock, self).__init__()
        self.entity = nn.Sequential()
        self.entity.add_module("conv", nn.ConvTranspose2d(ic, oc, ks, st,pd, bias=False))
        if bcnm:
            self.entity.add_module("bcnm", nn.BatchNorm2d(oc))
        self.entity.add_module("actv", nn.ReLU() if relu else nn.Tanh())

    def forward(self, x):
        return self.entity(x)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.entity = nn.Sequential(
            GeneratorBlock(100, 512, 4, 1, 0),
            GeneratorBlock(512, 256, 4, 2, 1),
            GeneratorBlock(256, 128, 4, 2, 1),
            GeneratorBlock(128,  64, 4, 2, 1),
            GeneratorBlock(64,    3, 4, 2, 1, bcnm=False, relu=False),
        )

    def forward(self, x):
        return self.entity(x)


generator = Generator().to(device).apply(init_weights)
