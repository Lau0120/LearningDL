from torch.utils.data import DataLoader
from torch import nn
from torch import optim
from torchvision import datasets, transforms

from train import training_loop
from model import VisualTransformer


cifar_100 = datasets.CIFAR100(
    root="../../Datasets/CIFAR100",
    train=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
    ]),
    download=False,
)
cifar_100_loader_trn = DataLoader(cifar_100, 128, True)


vit = VisualTransformer(
    n_block=12,
    n_tokens=768,
    patch_size=4,
    n_sequence=64,
    n_head=12,
    hidden_dim=3072,
    k=100,
    ratio=0.5,
).cuda()


training_loop(
    n_epochs=7,
    net=vit,
    loss_fn=nn.CrossEntropyLoss(),
    loader=cifar_100_loader_trn,
    optim=optim.Adam(vit.parameters(), lr=0.001),
)
