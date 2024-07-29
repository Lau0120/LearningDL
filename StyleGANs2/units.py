import torch
import torch.nn as nn
import torch.nn.functional as F


class EqualLinear(nn.Module):
    def __init__(self, ic, oc, lr_mul=1.):
        super(EqualLinear, self).__init__()
        self.weight = nn.Parameter(torch.randn(oc, ic))
        self.lr_mul = lr_mul
        self.bias = nn.Parameter(torch.randn(oc))

    def forward(self, x):
        return F.linear(x, self.weight * self.lr_mul, self.bias * self.lr_mul)


class Conv2dMod(nn.Module):
    def __init__(self, ic, oc, ks):
        super(Conv2dMod, self).__init__()
        self.o_channels = oc
        self.kernel_size = ks
        self.epsilon = 1e-8
        self.weight = nn.Parameter(torch.randn(oc, ic, ks, ks))

    def forward(self, x, style_code):
        b, c, h, w = x.shape

        # Modulation
        s = style_code[:, None, :, None, None]        # [b,  _, ic, _, _]
        expand_weight = self.weight[None, :, :, :, :] # [_, oc, ic, k, k]
        w1 = expand_weight * s

        # Demodulation
        sigma = torch.rsqrt((w1 ** 2).sum(dim=(2, 3, 4), keepdim=True) + self.epsilon)
        w2 = w1 * sigma

        # Adjust Shape                                           v
        x = x.reshape(1, -1, h, w)                         # [1, b*ic, h, w]
        _, _, *weight_dims = w2.shape                      #     v    [ic, h, w]
        w2 = w2.reshape(b * self.o_channels, *weight_dims) #    [b*oc, ic, h, w]

        # Apply Group Convolution
        x = F.conv2d(x, w2, padding="same", groups=b)
        x = x.reshape(-1, self.o_channels, h, w)           # [b, oc, h, w]

        return x


class GenerateRGB(nn.Module):
    def __init__(self, ic):
        super(GenerateRGB, self).__init__()
        self.to_rgb = nn.Conv2d(ic, 3, 3, 1, 1)
        self.up_conv = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, prev_rgb):
        next_rgb = self.to_rgb(x)
        rgb = prev_rgb + next_rgb
        return self.up_conv(rgb)


class ToStyle(nn.Module):
    def __init__(self, ic, oc):
        super(ToStyle, self).__init__()
        self.translator = nn.Linear(ic, oc)

    def forward(self, w):
        return self.translator(w)


class NoiseInjector(nn.Module):
    def __init__(self, ic):
        super(NoiseInjector, self).__init__()
        self.scaler = nn.Parameter(torch.randn([1, ic, 1, 1]))

    def forward(self, x):
        noise = torch.randn(x.shape, device=x.device)
        return x + torch.mul(noise, self.scaler)


class ActivationFactory:
    @staticmethod
    def LeakyRelu():
        return nn.LeakyReLU(negative_slope=0.2, inplace=True)

    @staticmethod
    def Tanh():
        return nn.Tanh()

    @staticmethod
    def ReLU():
        return nn.ReLU(inplace=True)
