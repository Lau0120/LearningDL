import torch
import torch.nn as nn

from units import EqualLinear, ActivationFactory
from blocks import SynthesisBlock, DownSampleResBlock


class MappingNetwork(nn.Module):
    def __init__(self, z_dim, depth=8, lr_mul=0.1):
        super(MappingNetwork, self).__init__()
        layers = []
        for _ in range(depth):
            layers.extend([EqualLinear(z_dim, z_dim, lr_mul), ActivationFactory.LeakyRelu()])
        self.entity = nn.Sequential(*layers)

    def forward(self, z):
        return self.entity(z)


class SynthesisNetwork(nn.Module):
    def __init__(self):
        super(SynthesisNetwork, self).__init__()
        self.constant = nn.Parameter(torch.randn((128, 4, 4)))
        self.layer1 = SynthesisBlock(128, 64)
        self.layer2 = SynthesisBlock( 64, 32)
        self.layer3 = SynthesisBlock( 32, 16)
        self.layer4 = SynthesisBlock( 16,  3)

    def forward(self, w):
        b, _ = w.shape
        x = torch.stack([self.constant] * b, 0)
        rgb = torch.zeros(b, 3, 8, 8).cuda()
        x, rgb = self.layer1(x, w, rgb)
        x, rgb = self.layer2(x, w, rgb)
        x, rgb = self.layer3(x, w, rgb)
        x, rgb = self.layer4(x, w, rgb)
        return x, rgb


class StyleGenerator(nn.Module):
    def __init__(self):
        super(StyleGenerator, self).__init__()
        self.map = MappingNetwork(512)
        self.synthesis = SynthesisNetwork()

    def forward(self, z):
        w = self.map(z)
        x = self.synthesis(w)
        return x


class StyleDiscriminator(nn.Module):
    def __init__(self):
        super(StyleDiscriminator, self).__init__()
        self.layer1 = DownSampleResBlock(  3,  32)
        self.layer2 = DownSampleResBlock( 32,  64)
        self.layer3 = DownSampleResBlock( 64, 128)
        self.layer4 = DownSampleResBlock(128, 256)
        self.layer5 = DownSampleResBlock(256, 512)
        self.layer6 = DownSampleResBlock(512, 512)
        self.layer7 = DownSampleResBlock(512,   1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        return x.view(-1, 1)
