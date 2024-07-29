import torch
import torch.nn.functional as F

from torch import nn

from components import ConvUnit, FlcnUnit, EasyPointNet, EdgeBlock
from utils import calc_prior_matrix, adain


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.share = EasyPointNet()
        self.mlp = nn.Sequential()
        self.mlp.add_module("flcn1", FlcnUnit(1024, 512, bc=False, lk=True))
        self.mlp.add_module("flcn2", FlcnUnit( 512, 256, bc=False, lk=True))
        self.mlp.add_module("flcn3", FlcnUnit( 256,  64, bc=False, lk=True))
        self.mlp.add_module("flcn4", FlcnUnit(  64,   1, bc=False, rl=False))

    def forward(self, x):
        return self.mlp(self.share(x))


class Generator(nn.Module):
    def __init__(self, noise_dim):
        super(Generator, self).__init__()
        self.noise_dim = noise_dim
        # * embedding feat 
        self.feat_embed_block = nn.Sequential(
            ConvUnit(3 + noise_dim, 128, 1, bc=False, lk=True),
            ConvUnit(          128, 128, 1, bc=False, lk=True),
        )
        # * embedding styl
        self.fst_styl_embed = ConvUnit(128, 128, 1, bc=False, rl=False)
        self.snd_styl_embed = ConvUnit(128, 256, 1, bc=False, rl=False)
        # * instance norms
        self.fst_inst_norms = nn.InstanceNorm1d( 64)
        self.snd_inst_norms = nn.InstanceNorm1d(128)
        # * graph attention module
        self.fst_gam = EdgeBlock( 3,  64, 20)
        self.snd_gam = EdgeBlock(64, 128, 20)
        # * global
        self.global_handler = nn.Sequential(
            FlcnUnit(128, 128, lk=True),
            FlcnUnit(128, 512, lk=True),
        )
        # * final
        self.final_handler = nn.Sequential(
            ConvUnit(640, 256, 1, bc=False, lk=True),
            ConvUnit(256,  64, 1, bc=False, lk=True),
            ConvUnit( 64,   3, 1, bc=False, rl=False, th=True),
        )
        # * tools
        self.fst_lkrl = nn.LeakyReLU(0.2)
        self.snd_lkrl = nn.LeakyReLU(0.2)

    def forward(self, x):
        b, _, n = x.shape
        # * prior matrix
        prior_matrix = calc_prior_matrix(x, self.noise_dim)
        # * embeddings
        feats = self.feat_embed_block(prior_matrix)
        fst_styl = self.fst_styl_embed(feats)
        snd_styl = self.snd_styl_embed(feats)
        # * fst process
        fst_global_feats = self.fst_inst_norms(self.fst_lkrl(self.fst_gam(x)))
        x = adain(fst_global_feats, fst_styl)
        # * snd process
        snd_global_feats = self.snd_inst_norms(self.snd_lkrl(self.snd_gam(x)))
        x = adain(snd_global_feats, snd_styl)
        # * global
        global_feats = F.max_pool1d(x, kernel_size=n).view(b, -1)
        global_feats = self.global_handler(global_feats).unsqueeze(2).expand(-1, -1, n)
        real_fianl = torch.concat((x, global_feats), dim=1)
        # * final
        return self.final_handler(real_fianl)
