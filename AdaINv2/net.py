import torch.nn as nn
import torch.nn.functional as F

from collections import namedtuple
from torchvision import models

from components import VGGDecoderBlock
from utils import AdaIN


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.block1 = VGGDecoderBlock(512, 256, 1)
        self.block2 = VGGDecoderBlock(256, 128, 4)
        self.block3 = VGGDecoderBlock(128,  64, 2)
        self.block4 = VGGDecoderBlock( 64,   3, 2, up=False, last_relu=False)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        return x

# class Decoder(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
#
#         block1 = nn.Sequential(
#             nn.Conv2d(512, 256, 3, 1, 1, padding_mode="reflect"), nn.ReLU()
#         )
#
#         block2 = nn.Sequential(
#             nn.Conv2d(256, 256, 3, 1, 1, padding_mode="reflect"),
#             nn.ReLU(),
#             nn.Conv2d(256, 256, 3, 1, 1, padding_mode="reflect"),
#             nn.ReLU(),
#             nn.Conv2d(256, 256, 3, 1, 1, padding_mode="reflect"),
#             nn.ReLU(),
#             nn.Conv2d(256, 128, 3, 1, 1, padding_mode="reflect"),
#             nn.ReLU(),
#         )
#
#         block3 = nn.Sequential(
#             nn.Conv2d(128, 128, 3, 1, 1, padding_mode="reflect"),
#             nn.ReLU(),
#             nn.Conv2d(128, 64, 3, 1, 1, padding_mode="reflect"),
#             nn.ReLU(),
#         )
#
#         block4 = nn.Sequential(
#             nn.Conv2d(64, 64, 3, 1, 1, padding_mode="reflect"),
#             nn.ReLU(),
#             nn.Conv2d(64, 3, 3, 1, 1, padding_mode="reflect"),
#         )
#
#         self.net = nn.ModuleList([block1, block2, block3, block4])
#
#     def forward(self, x):
#         for ix, module in enumerate(self.net):
#             x = module(x)
#             # * upsample
#             if ix < len(self.net) - 1:
#                 x = F.interpolate(x, scale_factor=2, mode="nearest")
#         return x


class Encoder(nn.Module):
    def __init__(self, requires_grad=False):
        super(Encoder, self).__init__()
        # ! load pretrained VGG19
        vgg_pretrained_features = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features.eval()
        # ! split VGG16 into 4 slices for feature maps
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        # ! extract layers from pretrained VGG16
        for x in range( 0,  2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range( 2,  7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range( 7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        # ! close auto-close
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        # ! calc & save feature maps
        x = self.slice1(x)
        relu1_1 = x
        x = self.slice2(x)
        relu2_1 = x
        x = self.slice3(x)
        relu3_1 = x
        x = self.slice4(x)
        relu4_1 = x

        return (relu1_1, relu2_1, relu3_1, relu4_1)


# class Encoder(nn.Module):
#     def __init__(self, pretrained=True, requires_grad=False) -> None:
#         super().__init__()
#         vgg = models.vgg19(pretrained=pretrained).features
#         # * block1: conv1_1, relu1_1,
#         self.block1 = vgg[:2]
#         # * block2: conv1_2, relu1_2, conv2_1, relu2_1
#         self.block2 = vgg[2:7]
#         # * block3: conv2_2, relu2_2, conv3_1, relu3_1
#         self.block3 = vgg[7:12]
#         # * block4
#         self.block4 = vgg[12:21]
#
#         self.__set_grad(requires_grad)
#
#     def __set_grad(self, requires_grad: bool):
#         for p in self.parameters():
#             p.requires_grad = requires_grad
#
#     def forward(self, x, return_last=False):
#         f1 = self.block1(x)
#         f2 = self.block2(f1)
#         f3 = self.block3(f2)
#         f4 = self.block4(f3)
#
#         return f4 if return_last else (f1, f2, f3, f4)
