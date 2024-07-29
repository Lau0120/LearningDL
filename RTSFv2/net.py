import torch
import torch.nn as nn

from collections import namedtuple
from torchvision import models
from components import ConvLayer, ResidualBlock, UpsampleConvLayer


class StyleTransferNet(nn.Module):
    def __init__(self):
        super(StyleTransferNet, self).__init__()
        # ! init convolution layers
        self.conv1 = ConvLayer(  3, 32, ks=9, st=1)
        self.in1 = nn.InstanceNorm2d(32, affine=True)
        self.conv2 = ConvLayer( 32, 64, ks=3, st=2)
        self.in2 = nn.InstanceNorm2d(64, affine=True)
        self.conv3 = ConvLayer(64, 128, ks=3, st=2)
        self.in3 = nn.InstanceNorm2d(128, affine=True)
        # ! residual layers
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)
        # ! upsampling layers
        self.deconv1 = UpsampleConvLayer(128, 64, ks=3, st=1, up=2)
        self.in4 = nn.InstanceNorm2d(64, affine=True)
        self.deconv2 = UpsampleConvLayer( 64, 32, ks=3, st=1, up=2)
        self.in5 = nn.InstanceNorm2d(32, affine=True)
        self.deconv3 = UpsampleConvLayer( 32,  3, ks=9, st=1)
        # ! non-linearities
        self.relu = nn.ReLU()

    def forward(self, x):
        # ! forward pass
        x = self.relu(self.in1(self.conv1(x)))
        x = self.relu(self.in2(self.conv2(x)))
        x = self.relu(self.in3(self.conv3(x)))
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = self.relu(self.in4(self.deconv1(x)))
        x = self.relu(self.in5(self.deconv2(x)))
        x = self.deconv3(x)

        return x


class Vgg16(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg16, self).__init__()
        # ! load pretrained VGG16
        vgg_pretrained_features = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features
        # ! split VGG16 into 4 slices for feature maps
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        # ! extract layers from pretrained VGG16
        for x in range( 0,  4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range( 4,  9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range( 9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        # ! close auto-close
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        # ! calc & save feature maps
        x = self.slice1(x)
        relu1_2 = x
        x = self.slice2(x)
        relu2_2 = x
        x = self.slice3(x)
        relu3_3 = x
        x = self.slice4(x)
        relu4_3 = x
        layer_features = namedtuple("LayerFeatures", ["relu1_2", "relu2_2", "relu3_3", "relu4_3"])

        return layer_features(relu1_2, relu2_2, relu3_3, relu4_3)
