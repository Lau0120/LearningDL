import torch.nn as nn
import torch.nn.functional as F


class VGGDecoderUnit(nn.Module):
    def __init__(self, ic, oc, ks=3, st=1, relu=True):
        super().__init__()
        self.entity = nn.Sequential()
        self.entity.add_module("rfpd", nn.ReflectionPad2d(ks // 2))
        self.entity.add_module("conv", nn.Conv2d(ic, oc, ks, st))
        if relu:
            self.entity.add_module("relu", nn.ReLU())

    def forward(self, x):
        return self.entity(x)


class VGGDecoderBlock(nn.Module):
    def __init__(self, ic, oc, count, ks=3, st=1, up=True, last_relu=True):
        super().__init__()
        self.entity = nn.Sequential()
        for i in range(count - 1):
            self.entity.add_module("unit_body{}".format(i), VGGDecoderUnit(ic, ic))
        self.entity.add_module("unit_taile", VGGDecoderUnit(ic, oc, relu=last_relu))
        if up:
            self.entity.add_module("last_upsmp", nn.Upsample(scale_factor=2, mode="nearest"))

    def forward(self, x):
        return self.entity(x)
