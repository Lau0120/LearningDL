import torch.nn as nn
import math

from units import Conv2dMod, ToStyle, NoiseInjector, GenerateRGB, ActivationFactory


class SynthesisBlock(nn.Module):
    def __init__(self, ic, oc, z_dim=512):
        super(SynthesisBlock, self).__init__()
        self.up_sample = nn.UpsamplingBilinear2d(scale_factor=2)

        self.to_style1 = ToStyle(z_dim, ic)
        self.conv1 = Conv2dMod(ic, oc, 3)
        self.noise_injector1 = NoiseInjector(oc)

        self.to_style2 = ToStyle(z_dim, oc)
        self.conv2 = Conv2dMod(oc, oc, 3)
        self.noise_injector2 = NoiseInjector(oc)

        self.generate_rgb = GenerateRGB(oc)
        self.activation = ActivationFactory.LeakyRelu()

    def forward(self, x, w, prev_rgb):
        x = self.up_sample(x)

        x = self.conv1(x, self.to_style1(w))
        x = self.noise_injector1(x)
        x = self.activation(x)

        x = self.conv2(x, self.to_style2(w))
        x = self.noise_injector2(x)
        x = self.activation(x)

        rgb = self.generate_rgb(x, prev_rgb)

        return x, rgb


class DownSampleResBlock(nn.Module):
    def __init__(self, ic, oc):
        super(DownSampleResBlock, self).__init__()
        self.res_conv = nn.Conv2d(ic, oc, 1, 2, 0)

        self.conv = nn.Sequential(
            nn.Conv2d(ic, oc, 3, 1, 1),
            ActivationFactory.LeakyRelu(),
            nn.Conv2d(oc, oc, 3, 1, 1),
            ActivationFactory.LeakyRelu(),
        )

        self.down_sample = nn.Conv2d(oc, oc, 4, 2, 1)

    def forward(self, x):
        identity = self.res_conv(x)
        x = self.down_sample(self.conv(x))
        return (x + identity) * (1 / math.sqrt(2))
