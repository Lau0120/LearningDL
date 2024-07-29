import torch.nn as nn


class ConvLayer(nn.Module):
    def __init__(self, ic, oc, ks, st):
        super(ConvLayer, self).__init__()
        # ! pad
        self.rfpd = nn.ReflectionPad2d(ks // 2)
        # ! conv
        self.conv = nn.Conv2d(ic, oc, ks, st)

    def forward(self, x):
        return self.conv(self.rfpd(x))


class ResidualBlock(nn.Module):
    def __init__(self, c):
        super(ResidualBlock, self).__init__()
        # ! conv
        self.conv1 = ConvLayer(c, c, ks=3, st=1)
        # ! instance
        self.in1 = nn.InstanceNorm2d(c, affine=True)
        # ! conv
        self.conv2 = ConvLayer(c, c, ks=3, st=1)
        # ! instance
        self.in2 = nn.InstanceNorm2d(c, affine=True)
        # ! non-linear
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x

        x = self.in1(self.conv1(x))
        x = self.relu(x)
        x = self.in2(self.conv2(x))

        return x + identity


class UpsampleConvLayer(nn.Module):
    def __init__(self, ic, oc, ks, st, up=None):
        super(UpsampleConvLayer, self).__init__()
        # ! up-sample
        self.upsample = up
        # ! conv
        self.conv = ConvLayer(ic, oc, ks, st)

    def forward(self, x):
        if self.upsample is not None:
            x = nn.functional.interpolate(x, mode="nearest", scale_factor=self.upsample)
        return self.conv(x)
