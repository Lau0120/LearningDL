import torch
import torch.nn.functional as F

from torch import nn

from utils import sample_group, sample_group_all


class ConvUnit(nn.Module):
    def __init__(self, ic, oc, ks=1, st=1, pd=0, rl_required=True):
        super(ConvUnit, self).__init__()
        self.entity = nn.Sequential()
        self.entity.add_module("conv", nn.Conv1d(ic, oc, ks, st, pd))
        self.entity.add_module("bcnm", nn.BatchNorm1d(oc))
        if rl_required:
            self.entity.add_module("relu", nn.ReLU())
    
    def forward(self, x):
        return self.entity(x)


class FlcnUnit(nn.Module):
    def __init__(self, ic, oc, bc_required=True, dp_required=False, rl_required=True):
        super(FlcnUnit, self).__init__()
        self.entity = nn.Sequential()
        self.entity.add_module("flcn", nn.Linear(ic, oc))
        if bc_required:
            self.entity.add_module("bcnm", nn.BatchNorm1d(oc))
        if dp_required:
            self.entity.add_module("drop", nn.Dropout(0.3))
        if rl_required:
            self.entity.add_module("relu", nn.ReLU())
    
    def forward(self, x):
        return self.entity(x)


class Transform(nn.Module):
    def __init__(self, k):
        super(Transform, self).__init__()

        self.conv_entity = nn.Sequential()
        self.conv_entity.add_module("uConv1", ConvUnit(   k,   64))
        self.conv_entity.add_module("uConv2", ConvUnit(  64,  128))
        self.conv_entity.add_module("uConv3", ConvUnit( 128, 1024))

        self.flcn_entity = nn.Sequential()
        self.flcn_entity.add_module("uFlcn1", FlcnUnit(1024,  512))
        self.flcn_entity.add_module("uFlcn2", FlcnUnit( 512,  256))

        self.last = nn.Linear(256, k * k)
        self.k = k
    
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
        self.fst_mlp = ConvUnit(3, 64)
        self.feat_trans = Transform(k=64)
        self.snd_mlp = nn.Sequential()
        self.snd_mlp.add_module("conv1", ConvUnit( 64,  128))
        self.snd_mlp.add_module("conv2", ConvUnit(128, 1024, rl_required=False))

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
        return F.max_pool1d(x, kernel_size=x.shape[2]).view(x.shape[0], -1)


class DynamicMLP(nn.Module):
    def __init__(self, ic, ocs, ks, bc_required=True, rl_required=True):
        super(DynamicMLP, self).__init__()
        last_oc = ic
        self.entity = nn.Sequential()
        for i, oc in enumerate(ocs):
            self.entity.add_module("conv{}".format(i + 1), nn.Conv2d(last_oc, oc, ks))
            if bc_required:
                self.entity.add_module("bcnm{}".format(i + 1), nn.BatchNorm2d(oc))
            if rl_required:
                self.entity.add_module("relu{}".format(i + 1), nn.ReLU())
            last_oc = oc
    
    def forward(self, x):
        return self.entity(x)


class SetAbstractionLevel(nn.Module):
    def __init__(self, n_group, b_radius, k, ic, mlp, group_all):
        super(SetAbstractionLevel, self).__init__()
        self.n_group = n_group
        self.radius = b_radius
        self.n_sample = k
        self.group_all = group_all
        self.easy_mlp = DynamicMLP(ic, mlp, ks=1)
    
    def forward(self, xyz, points):
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)
        
        # * sample & group
        if self.group_all:
            new_xyz, new_points = sample_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_group(self.n_group, self.radius, self.n_sample, xyz, points)
        
        # * encoder
        new_points = new_points.permute(0, 3, 2, 1)
        new_points = self.easy_mlp(new_points)
        
        # * post-process
        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        
        return new_xyz, new_points
