import torch

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import VisualTransformer


cifar_100 = datasets.CIFAR100(
    root="../../Datasets/CIFAR100",
    train=False,
    transform=transforms.Compose([
        transforms.ToTensor(),
    ]),
    download=False,
)
cifar_100_loader_eval = DataLoader(cifar_100, 128, True)


vit = VisualTransformer(
    n_block=12,
    n_tokens=768,
    patch_size=4,
    n_sequence=64,
    n_head=12,
    hidden_dim=3072,
    k=100,
    ratio=0.5,
)
vit.load_state_dict(torch.load("./ckpts/classifier_{}.pth".format(8)))
vit.eval().cuda()


correct = 0
total = 0
with torch.no_grad():
    for imgs, lbes in cifar_100_loader_eval:
        imgs = imgs.to("cuda", dtype=torch.float)
        lbes = lbes.to("cuda")
        output = vit(imgs)
        _, predicted = torch.max(output, dim=1)
        total = total + lbes.shape[0]
        correct = correct + (predicted == lbes).sum()
print("Accuracy: {:.4f}".format(correct / total))
