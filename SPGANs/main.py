import torch
import numpy as np

from torch import nn, optim
from torch.utils.data import DataLoader

from model import Generator, Discriminator
from train import training_loop
from dataset import ShapeNet


shape_n = ShapeNet(
    path="../../Datasets/ShapeNetCore/ShapeNet_Chair",
    is_train=True,
    n_points=2048,
)
shape_loader = DataLoader(shape_n, 8, True)
sphere_pts = np.loadtxt("./balls/2048.xyz", usecols=(0, 1, 2))
sphere = torch.from_numpy(sphere_pts).transpose(0, 1)


gene = Generator(noise_dim=128).cuda()
disc = Discriminator().cuda()
training_loop(
    n_epochs=2000,
    loader=shape_loader,
    sphere=sphere,
    generator=gene,
    gene_optim=optim.Adam(gene.parameters(), lr=0.0001),
    discriminator=disc,
    disc_optim=optim.Adam(disc.parameters(), lr=0.0001),
    loss_fn=nn.MSELoss(),
)
