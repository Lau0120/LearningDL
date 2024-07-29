from torch import nn

from components import PointNet, FlcnUnit, SetAbstractionLevel


class ClaPointNet(nn.Module):
    def __init__(self, k):
        super(ClaPointNet, self).__init__()
        self.share = PointNet()
        self.flcn1 = FlcnUnit(1024, 512)
        self.flcn2 = FlcnUnit( 512, 256, dp_required=True)
        self.flcn3 = FlcnUnit( 256,   k, bc_required=False, rl_required=False)
    
    def forward(self, x):
        x = self.share(x)
        x = self.flcn1(x)
        x = self.flcn2(x)
        x = self.flcn3(x)

        return x


class ClaPointNet2SSG(nn.Module):
    def __init__(self, k):
        super(ClaPointNet2SSG, self).__init__()
        self.sale1 = SetAbstractionLevel(n_group= 512, b_radius= 0.2, k=  32, ic=      3,
                                         mlp=[ 64,  64,  128], group_all=False)
        self.sale2 = SetAbstractionLevel(n_group= 128, b_radius= 0.4, k=  64, ic=128 + 3,
                                         mlp=[128, 128,  256], group_all=False)
        self.sale3 = SetAbstractionLevel(n_group=None, b_radius=None, k=None, ic=256 + 3,
                                         mlp=[256, 512, 1024], group_all=True)
        self.flcn1 = FlcnUnit(1024, 512, dp_required=True)
        self.flcn2 = FlcnUnit( 512, 256, dp_required=True)
        self.flcn3 = FlcnUnit( 256,   k, bc_required=False, rl_required=False)

    def forward(self, x):
        b, _, _ = x.shape

        # * encoder 
        x, l1pt = self.sale1(x, None)
        x, l2pt = self.sale2(x, l1pt)
        x, l3pt = self.sale3(x, l2pt)

        # * decoder
        x = l3pt.view(b, 1024)
        x = self.flcn1(x)
        x = self.flcn2(x)
        x = self.flcn3(x)

        return x
