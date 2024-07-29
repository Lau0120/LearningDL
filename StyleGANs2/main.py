from torch import optim
from torchvision import transforms
from torch.utils.data import DataLoader

from dataset import CeleaHQ
from model import StyleGenerator, StyleDiscriminator
from train import training_loop


celeba_hq = CeleaHQ(
    path="../../Datasets/CelebA_HQ/celeba_hq/train",
    transform=transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ]),
)
celeba_hq_loader = DataLoader(celeba_hq, 32, True)


gene = StyleGenerator().train().cuda()
disc = StyleDiscriminator().train().cuda()
training_loop(
    n_epochs=2000,
    loader=celeba_hq_loader,
    gene=gene,
    gene_optim=optim.Adam(gene.parameters(), lr=1e-4, betas=(0.5, 0.999)),
    disc=disc,
    disc_optim=optim.Adam(disc.parameters(), lr=1e-4, betas=(0.5, 0.999)),
)
