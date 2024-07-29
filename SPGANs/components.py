import torch
import torch.nn.functional as F

from torch import nn

from utils import get_edge_features


class ConvUnit(nn.Module):
    def __init__(self, ic, oc, ks=1, st=1, pd=0, bc=True, rl=True, lk=False, th=False):
        super(ConvUnit, self).__init__()
        self.entity = nn.Sequential()
        self.entity.add_module("conv", nn.Conv1d(ic, oc, ks, st, pd))
        if bc:
            self.entity.add_module("bcnm", nn.BatchNorm1d(oc))
        if rl:
            self.entity.add_module("relu", nn.LeakyReLU(inplace=True) if lk else nn.ReLU())
        if th:
            self.entity.add_module("tanh", nn.Tanh())
    
    def forward(self, x):
        return self.entity(x)


class FlcnUnit(nn.Module):
    def __init__(self, ic, oc, bc=True, dp=False, rl=True, lk=False):
        super(FlcnUnit, self).__init__()
        self.entity = nn.Sequential()
        self.entity.add_module("flcn", nn.Linear(ic, oc))
        if bc:
            self.entity.add_module("bcnm", nn.BatchNorm1d(oc))
        if dp:
            self.entity.add_module("drop", nn.Dropout(0.3))
        if rl:
            self.entity.add_module("relu", nn.LeakyReLU(inplace=True) if lk else nn.ReLU())
    
    def forward(self, x):
        return self.entity(x)


class Transform(nn.Module):
    def __init__(self, k):
        super(Transform, self).__init__()
        self.conv_entity = nn.Sequential()
        self.conv_entity.add_module("uConv1", ConvUnit(   k,   64, lk=True))
        self.conv_entity.add_module("uConv2", ConvUnit(  64,  128, lk=True))
        self.conv_entity.add_module("uConv3", ConvUnit( 128, 1024, lk=True))
        self.flcn_entity = nn.Sequential()
        self.flcn_entity.add_module("uFlcn1", FlcnUnit(1024,  512, lk=True))
        self.flcn_entity.add_module("uFlcn2", FlcnUnit( 512,  256, lk=True))
        self.k = k
        self.last = nn.Linear(256, self.k * self.k)
    
    def forward(self, x):
        x = self.conv_entity(x)
        _, channel, count = x.shape
        x = F.max_pool1d(x, kernel_size=count).view(-1, channel)
        x = self.flcn_entity(x)
        x = self.last(x)
        return x.view(-1, self.k, self.k)


class PointNet(nn.Module):
    def __init__(self):
        super(PointNet, self).__init__()
        self.inpu_trans = Transform(k= 3)
        self.fst_mlp = ConvUnit(3, 64, lk=True)
        self.feat_trans = Transform(k=64)
        self.snd_mlp = nn.Sequential()
        self.snd_mlp.add_module("conv1", ConvUnit( 64,  128, lk=True))
        self.snd_mlp.add_module("conv2", ConvUnit(128, 1024, lk=True))

    def forward(self, x):
        trans = self.inpu_trans(x)
        x = x.transpose(1, 2)
        x = torch.bmm(x, trans)
        x = x.transpose(1, 2)
        x = self.fst_mlp(x)
        trans = self.feat_trans(x)
        x = x.transpose(1, 2)
        x = torch.bmm(x, trans)
        x = x.transpose(1, 2)
        x = self.snd_mlp(x)
        return x, F.max_pool1d(x, kernel_size=x.shape[2])


class EasyPointNet(nn.Module):
    def __init__(self):
        super(EasyPointNet, self).__init__()
        self.entity = nn.Sequential()
        self.entity.add_module("conv1", ConvUnit(  3,   64, lk=True))
        self.entity.add_module("conv2", ConvUnit( 64,  128, lk=True))
        self.entity.add_module("conv3", ConvUnit(128,  256, lk=True))
        self.entity.add_module("conv4", ConvUnit(256, 1024, lk=True))
    
    def forward(self, x):
        b, _, _ = x.shape
        x = self.entity(x)
        x = F.adaptive_max_pool1d(x, 1).view(b, -1)
        return x


class EdgeBlock(nn.Module):
    def __init__(self, ic, oc, k):
        super(EdgeBlock, self).__init__()
        self.k = k
        self.conv_w = nn.Sequential(
            nn.Conv2d(ic, oc // 2, 1),
            nn.BatchNorm2d(oc // 2),
            nn.LeakyReLU(),
            nn.Conv2d(oc // 2, oc, 1),
            nn.BatchNorm2d(oc),
            nn.LeakyReLU()
        )
        self.conv_x = nn.Sequential(
            nn.Conv2d(2 * ic, oc, [1, 1], [1, 1]),
            nn.BatchNorm2d(oc),
            nn.LeakyReLU()
        )
        self.conv_out = nn.Conv2d(oc, oc, [1, k], [1, 1])

    def forward(self, x):
        _, c, _ = x.shape
        x = get_edge_features(x, self.k)
        w = self.conv_w(x[:, c:, :, :])
        w = F.softmax(w, dim=-1)
        x = self.conv_x(x)
        x = x * w
        x = self.conv_out(x)
        x = x.squeeze(3)
        return x
