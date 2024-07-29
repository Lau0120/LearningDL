import torch
import torch.optim as optim
import albumentations.augmentations as A
import numpy as np
import random

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from train import training_loop
from net import Encoder, Decoder


class ResizeShortest:
    def __init__(self, size=512) -> None:
        assert isinstance(size, (int, tuple))
        self.size = 512
        self.resize_tf = A.SmallestMaxSize(self.size)

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        img = self.resize_tf(image=np.array(image))
        return img["image"]


preprocess = transforms.Compose([
    ResizeShortest(512),
    transforms.ToTensor(),
    transforms.CenterCrop(256),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    ),
])


cocos = ImageFolder(
    root="G:/Project/Deep_Learning/Datasets/COCOs/train2014",
    transform=preprocess,
)
cocos_loader = DataLoader(cocos, 7, shuffle=True)


styls = ImageFolder(
    root="G:/Project/Deep_Learning/Datasets/PainterByNumbers/archive",
    transform=preprocess,
)
styls_loader = DataLoader(styls, 7, shuffle=True)


encoder = Encoder().to(torch.device("cuda"))
decoder = Decoder().to(torch.device("cuda"))
training_loop(
    n_epochs=5,
    cont_loader=cocos_loader,
    styl_loader=styls_loader,
    encoder=encoder,
    decoder=decoder,
    optimizer=optim.Adam(decoder.parameters(), lr=1e-4),
    styl_factor=10,
)
