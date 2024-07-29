import torch
import numpy as np

from model import Generator
from utils import visl_point_cloud_from_tensor


gene = Generator(noise_dim=128)
gene.load_state_dict(torch.load("./ckpts/plane/generator_v{}.pth".format(50)))
gene.eval().cuda()


sphere_pts = np.loadtxt("./balls/2048.xyz", usecols=(0, 1, 2))
sphere = torch.from_numpy(sphere_pts).transpose(0, 1).unsqueeze(0).to("cuda", dtype=torch.float)
output = gene(sphere)
visl_point_cloud_from_tensor(output)
