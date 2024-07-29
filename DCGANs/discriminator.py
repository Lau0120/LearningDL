import torch.nn as nn

from util import init_weights
from config import device


class DiscriminatorBlock(nn.Module):
    def __init__(self, ic, oc, ks, st, pd, bcnm=True, leaky_relu=True):
        super(DiscriminatorBlock, self).__init__()
        self.entity = nn.Sequential()
        self.entity.add_module("conv", nn.Conv2d(ic, oc, ks, st,pd, bias=False))
        if bcnm:
            self.entity.add_module("bcnm", nn.BatchNorm2d(oc))
        self.entity.add_module("actv", nn.LeakyReLU(0.2, True) if leaky_relu else nn.Sigmoid())

    def forward(self, x):
        return self.entity(x)


class Discriminator(nn.Module):
    def __init__(self) -> None:
        super(Discriminator, self).__init__()
        self.entity = nn.Sequential(
            DiscriminatorBlock(  3,  64, 4, 2, 1, bcnm=False),
            DiscriminatorBlock( 64, 128, 4, 2, 1),
            DiscriminatorBlock(128, 256, 4, 2, 1),
            DiscriminatorBlock(256, 512, 4, 2, 1),
            DiscriminatorBlock(512,   1, 4, 1, 0, bcnm=False, leaky_relu=False),
        )
    
    def forward(self, x):
        return self.entity(x)


discriminator = Discriminator().to(device).apply(init_weights)
